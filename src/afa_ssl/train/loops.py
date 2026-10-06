"""Training loops. Every loop uses a fixed number of gradient steps (not epochs)
so that label levels from 25 to 1,500 rows get the same optimisation budget."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from afa_ssl.data.masks import masks_tensor
from afa_ssl.models.dime import ValueNetwork, cmi_targets, rollout_masks
from afa_ssl.models.eddi import PartialVAE
from afa_ssl.models.mfa import MFA, AugClassifier, SSLClassifier, SupClassifier
from afa_ssl.utils.config import BaselineConfig, MaskConfig, ModelConfig, TrainConfig


def _batches(n: int, batch: int, rng: np.random.Generator):
    while True:
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            yield perm[s:s + batch]


def train_mfa(X: np.ndarray, valid: np.ndarray | None, mcfg: ModelConfig, tcfg: TrainConfig,
              kcfg: MaskConfig, rng: np.random.Generator, log=None, deadline_s: float | None = None
              ) -> tuple[MFA, dict]:
    """Self-supervised pretraining on rows X (features only). `valid` marks
    non-missing entries. A 10% validation slice picks the best checkpoint."""
    n, d = X.shape
    xt = torch.from_numpy(X)
    vt = torch.from_numpy(valid.astype(np.float32)) if valid is not None else None
    n_val = max(int(tcfg.ssl_val_frac * n), 16)
    perm = rng.permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    model = MFA(d, mcfg.hidden, mcfg.latent, mcfg.n_layers, mcfg.dropout,
                mcfg.logvar_min, mcfg.logvar_max, objective=mcfg.ssl_objective)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.ssl_lr, weight_decay=tcfg.ssl_wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, tcfg.ssl_steps)
    val_masks = masks_tensor(n_val, d, kcfg.budget_max, kcfg.budget_mix, rng)
    best, best_val, best_step = None, float("inf"), 0
    t0 = time.time()
    it = _batches(len(tr_idx), tcfg.ssl_batch, rng)
    hist = []
    for step in range(1, tcfg.ssl_steps + 1):
        idx = tr_idx[next(it)]
        m = masks_tensor(len(idx), d, kcfg.budget_max, kcfg.budget_mix, rng)
        loss = model.loss(xt[idx], m, None if vt is None else vt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % tcfg.ssl_eval_every == 0 or step == tcfg.ssl_steps:
            model.eval()
            with torch.no_grad():
                vl = model.loss(xt[val_idx], val_masks, None if vt is None else vt[val_idx]).item()
            model.train()
            hist.append({"step": step, "train_nll": loss.item(), "val_nll": vl})
            if log:
                log.event("ssl_step", step=step, train_nll=loss.item(), val_nll=vl,
                          elapsed=time.time() - t0)
            if vl < best_val:
                best_val, best_step, best = vl, step, copy.deepcopy(model.state_dict())
            if deadline_s is not None and time.time() - t0 > deadline_s:
                if log:
                    log.event("ssl_deadline", step=step)
                break
    model.load_state_dict(best)
    model.eval()
    if mcfg.ssl_objective == "mse":
        model.fit_fixed_logvar(xt[val_idx], val_masks, None if vt is None else vt[val_idx])
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"best_val_nll": best_val, "best_step": best_step, "history": hist,
                   "seconds": time.time() - t0}


def train_sup_classifier(X: np.ndarray, y: np.ndarray, n_classes: int, mcfg: ModelConfig,
                         tcfg: TrainConfig, kcfg: MaskConfig, rng: np.random.Generator, log=None,
                         valid: np.ndarray | None = None, pre: MFA | None = None,
                         sample_weight: np.ndarray | None = None):
    """From-scratch masked classifier. With `pre`, the frozen pretrained encoder's latent is an
    extra input of the first layer (AugClassifier); otherwise a plain SupClassifier."""
    n, d = X.shape
    xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    vt = torch.from_numpy(valid.astype(np.float32)) if valid is not None else None
    swt = torch.from_numpy(sample_weight.astype(np.float32)) if sample_weight is not None else None
    if pre is not None:
        model = AugClassifier(pre.encoder, d, mcfg.hidden, mcfg.latent, n_classes, mcfg.n_layers, mcfg.dropout)
        params = list(model.trainable_parameters())
    else:
        model = SupClassifier(d, mcfg.hidden, mcfg.latent, n_classes, mcfg.n_layers, mcfg.dropout)
        params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=tcfg.sup_lr, weight_decay=tcfg.sup_wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, tcfg.sup_steps)
    it = _batches(n, min(tcfg.sup_batch, n), rng)
    t0 = time.time()
    for step in range(1, tcfg.sup_steps + 1):
        idx = next(it)
        m = masks_tensor(len(idx), d, kcfg.budget_max, kcfg.budget_mix, rng)
        if vt is not None:
            m = m * vt[idx]
        if swt is None:
            loss = F.cross_entropy(model(xt[idx], m), yt[idx])
        else:
            loss = (F.cross_entropy(model(xt[idx], m), yt[idx], reduction="none") * swt[idx]).sum() / swt[idx].sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log and step % 500 == 0:
            log.event("sup_step", step=step, loss=loss.item())
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"final_loss": float(loss.item()), "seconds": time.time() - t0}


def train_head(mfa: MFA, X: np.ndarray, y: np.ndarray, n_classes: int, mcfg: ModelConfig,
               tcfg: TrainConfig, kcfg: MaskConfig, rng: np.random.Generator, log=None,
               valid: np.ndarray | None = None, X_unl: np.ndarray | None = None,
               valid_unl: np.ndarray | None = None, sample_weight: np.ndarray | None = None
               ) -> tuple[SSLClassifier, dict]:
    """Few-shot head on the pretrained encoder (frozen, or fine-tuned when tcfg.finetune),
    trained under the same mask distribution. With tcfg.distill_weight > 0 and unlabeled rows
    X_unl, the partial-view prediction on unlabeled rows is pulled towards the classifier's own
    (detached) full-view prediction: p(y | x_O) should look like the marginal of p(y | x)."""
    n, d = X.shape
    xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    vt = torch.from_numpy(valid.astype(np.float32)) if valid is not None else None
    swt = torch.from_numpy(sample_weight.astype(np.float32)) if sample_weight is not None else None
    distill = tcfg.distill_weight > 0 and X_unl is not None
    if distill:
        xu = torch.from_numpy(X_unl)
        vu = torch.from_numpy(valid_unl.astype(np.float32)) if valid_unl is not None else torch.ones_like(xu)
        it_u = _batches(len(xu), min(tcfg.distill_batch, len(xu)), rng)
    clf = SSLClassifier(mfa, n_classes, head=mcfg.head)
    if tcfg.finetune:
        # the classifier gets its own copy of the encoder; the MFA (and its conditional) stays frozen
        clf.encoder = copy.deepcopy(mfa.encoder)
        for p_ in clf.encoder.parameters():
            p_.requires_grad_(True)
        groups = [{"params": list(clf.head_parameters()), "lr": tcfg.head_lr},
                  {"params": clf.encoder.parameters(), "lr": tcfg.ft_lr_encoder}]
    else:
        groups = [{"params": list(clf.head_parameters()), "lr": tcfg.head_lr}]
    opt = torch.optim.Adam(groups, weight_decay=tcfg.head_wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, tcfg.head_steps)
    it = _batches(n, min(tcfg.head_batch, n), rng)
    t0 = time.time()
    for step in range(1, tcfg.head_steps + 1):
        idx = next(it)
        m = masks_tensor(len(idx), d, kcfg.budget_max, kcfg.budget_mix, rng)
        if vt is not None:
            m = m * vt[idx]
        if tcfg.finetune:
            z = clf.encoder(xt[idx], m)
        else:
            with torch.no_grad():
                z = clf.encoder(xt[idx], m)
        if swt is None:
            loss = F.cross_entropy(clf.head(z), yt[idx])
        else:
            loss = (F.cross_entropy(clf.head(z), yt[idx], reduction="none") * swt[idx]).sum() / swt[idx].sum()
        if distill:
            ui = next(it_u)
            with torch.no_grad():
                teacher = F.softmax(clf(xu[ui], vu[ui]), dim=-1)      # full view (all non-missing)
            mu_ = masks_tensor(len(ui), d, kcfg.budget_max, kcfg.budget_mix, rng) * vu[ui]
            student = F.log_softmax(clf(xu[ui], mu_), dim=-1)          # partial view
            loss = loss + tcfg.distill_weight * (-(teacher * student).sum(-1).mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log and step % 500 == 0:
            log.event("head_step", step=step, loss=loss.item())
    clf.eval()
    for p in clf.parameters():
        p.requires_grad_(False)
    return clf, {"final_loss": float(loss.item()), "seconds": time.time() - t0}


def train_partial_vae(X: np.ndarray, y: np.ndarray, has_label: np.ndarray, n_classes: int,
                      bcfg: BaselineConfig, tcfg: TrainConfig, kcfg: MaskConfig,
                      rng: np.random.Generator, log=None, valid: np.ndarray | None = None,
                      deadline_s: float | None = None) -> tuple[PartialVAE, dict]:
    """EDDI's partial VAE. Rows with has_label = 0 contribute only the feature terms of the
    ELBO, so the same code trains the labeled-only and the pool version."""
    n, d = X.shape
    xt = torch.from_numpy(X)
    yt = torch.from_numpy(np.maximum(y, 0))
    hl = torch.from_numpy(has_label.astype(np.float32))
    vt = torch.from_numpy(valid.astype(np.float32)) if valid is not None else None
    model = PartialVAE(d, n_classes, bcfg.pvae_emb, bcfg.pvae_hidden, bcfg.pvae_latent,
                       bcfg.pvae_latent_samples, label_weight=bcfg.pvae_label_weight)
    opt = torch.optim.Adam(model.parameters(), lr=bcfg.pvae_lr, weight_decay=bcfg.pvae_wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, bcfg.pvae_steps)
    n_val = max(int(tcfg.ssl_val_frac * n), 16)
    perm = rng.permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    val_masks = masks_tensor(n_val, d, kcfg.budget_max, kcfg.budget_mix, rng)
    best, best_val, best_step = None, float("inf"), 0
    it = _batches(len(tr_idx), min(bcfg.pvae_batch, len(tr_idx)), rng)
    t0 = time.time()
    hist = []
    for step in range(1, bcfg.pvae_steps + 1):
        idx = tr_idx[next(it)]
        m = masks_tensor(len(idx), d, kcfg.budget_max, kcfg.budget_mix, rng)
        if vt is not None:
            m = m * vt[idx]
        y_obs = torch.from_numpy((rng.random(len(idx)) < bcfg.pvae_label_obs_prob).astype(np.float32)) * hl[idx]
        loss = model.loss(xt[idx], m, yt[idx], y_obs, hl[idx], None if vt is None else vt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % bcfg.pvae_eval_every == 0 or step == bcfg.pvae_steps:
            model.eval()
            with torch.no_grad():
                vl = model.loss(xt[val_idx], val_masks, yt[val_idx], torch.zeros(n_val),
                                hl[val_idx], None if vt is None else vt[val_idx]).item()
            model.train()
            hist.append({"step": step, "train_elbo": loss.item(), "val_elbo": vl})
            if vl < best_val:
                best_val, best_step, best = vl, step, copy.deepcopy(model.state_dict())
            if log and step % (5 * bcfg.pvae_eval_every) == 0:
                log.event("pvae_step", step=step, train_elbo=loss.item(), val_elbo=vl)
            if deadline_s is not None and time.time() - t0 > deadline_s:
                if log:
                    log.event("pvae_deadline", step=step)
                break
    model.load_state_dict(best)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    return model, {"best_val_elbo": best_val, "best_step": best_step, "history": hist,
                   "seconds": time.time() - t0}


def train_dime_value(predictor: nn.Module, X: np.ndarray, y: np.ndarray, bcfg: BaselineConfig,
                     kcfg: MaskConfig, rng: np.random.Generator, log=None,
                     valid: np.ndarray | None = None) -> tuple[ValueNetwork, dict]:
    """Regress the value network on the realised cross-entropy reduction of the predictor,
    with masks drawn from the current greedy policy."""
    n, d = X.shape
    xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    vt = torch.from_numpy(valid.astype(np.float32)) if valid is not None else None
    net = ValueNetwork(d, bcfg.dime_hidden)
    opt = torch.optim.Adam(net.parameters(), lr=bcfg.dime_lr, weight_decay=bcfg.dime_wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, bcfg.dime_steps)
    it = _batches(n, min(bcfg.dime_batch, n), rng)
    t0 = time.time()
    loss = torch.tensor(0.0)
    for step in range(1, bcfg.dime_steps + 1):
        idx = next(it)
        if step <= bcfg.dime_steps // 4:     # warm start before the policy means anything
            m = masks_tensor(len(idx), d, kcfg.budget_max, kcfg.budget_mix, rng)
            if vt is not None:
                m = m * vt[idx]
        else:
            net.eval()
            m = rollout_masks(net, xt[idx], kcfg.budget_max, rng, None if vt is None else vt[idx])
            net.train()
        target = cmi_targets(predictor, xt[idx], yt[idx], m)
        pred = net(xt[idx], m)
        w = (1.0 - m) if vt is None else (1.0 - m) * vt[idx]
        loss = (((pred - target) ** 2) * w).sum() / w.sum().clamp_min(1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log and step % 500 == 0:
            log.event("dime_step", step=step, loss=loss.item())
    net.eval()
    for prm in net.parameters():
        prm.requires_grad_(False)
    return net, {"final_loss": float(loss.item()), "seconds": time.time() - t0}
