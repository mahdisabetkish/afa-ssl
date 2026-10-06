"""One run = (dataset, seed, n_labels). Trains the shared models once and
evaluates every requested acquisition method on the same test rows."""
from __future__ import annotations

import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from afa_ssl.acquisition.cost import make_costs
from afa_ssl.acquisition.policy import EpisodeResult, run_episode
from afa_ssl.data.loaders import build_dataset
from afa_ssl.data.splits import make_split, standardise
from afa_ssl.eval.metrics import (accuracy_curve, aucc_cost, aucc_uniform, group_coverage,
                                  precision_at_k, redundancy_rate)
from afa_ssl.models.conditional import GaussianConditional, MFAConditional
from afa_ssl.models.mfa import EnsembleClassifier, predict_proba
from afa_ssl.train.loops import (train_dime_value, train_head, train_mfa, train_partial_vae,
                                  train_sup_classifier)
from afa_ssl.utils.config import RunConfig, config_hash, save_config, to_dict
from afa_ssl.utils.io import append_registry, git_commit, write_json
from afa_ssl.utils.logging import RunLogger
from afa_ssl.utils.resources import machine_state
from afa_ssl.utils.seed import seed_everything

# name -> (classifier, conditional, criterion, cost_aware)
METHODS: dict[str, tuple[str, str | None, str, bool]] = {
    "rand": ("sup", None, "rand", True),
    "static_sup": ("sup", None, "static", True),
    "ssl_static": ("ssl", None, "static", True),
    "unc": ("sup", "gauss_lab", "unc", True),
    "eig_sup": ("sup", "gauss_lab", "eig", True),
    "er_sup": ("sup", "gauss_lab", "er", True),
    "ecd_sup": ("sup", "gauss_lab", "ecd", True),
    "eig_sup_mfa": ("sup", "mfa_lab", "eig", True),
    "ssl_rand": ("ssl", None, "rand", True),
    "ssl_unc": ("ssl", "mfa_pool", "unc", True),
    "ours": ("ens", "mfa_pool", "eig", True),            # SS-EIG: ensemble predictive + SSL conditional
    "ours_sslonly": ("ssl", "mfa_pool", "eig", True),    # ablation: pretrained member only
    "ours_supcond": ("ens", "gauss_lab", "eig", True),   # proposed: ensemble predictive + Gaussian conditional on labeled rows
    "ours_w05": ("ens05", "mfa_pool", "eig", True),      # ablation: fixed equal mixture weight
    "ens_rand": ("ens", None, "rand", True),             # same predictive, random order
    "ens_static": ("ens", None, "static", True),
    "ens_unc": ("ens", "mfa_pool", "unc", True),
    "ens_er": ("ens", "mfa_pool", "er", True),
    "ens_ecd": ("ens", "mfa_pool", "ecd", True),
    "eig_sup_ens": ("sup_ens", "gauss_lab", "eig", True),  # fair baseline: two from-scratch members
    "sup_ens_rand": ("sup_ens", None, "rand", True),
    "aug_eig": ("aug", "mfa_pool", "eig", True),
    "aug_rand": ("aug", None, "rand", True),
    "augens_eig": ("augens", "mfa_pool", "eig", True),
    "augens_rand": ("augens", None, "rand", True),
    # published baselines
    "eddi": ("pvae_lab", "pvae_lab", "ecd", True),
    "eddi_eig": ("pvae_lab", "pvae_lab", "eig", True),
    "eddi_rand": ("pvae_lab", None, "rand", True),
    "eddi_ssl": ("pvae_pool", "pvae_pool", "ecd", True),
    "eddi_ssl_eig": ("pvae_pool", "pvae_pool", "eig", True),
    "eddi_ssl_rand": ("pvae_pool", None, "rand", True),
    "dime": ("sup", None, "learned", True),
    # the proposed predictor dropped into other acquisition methods, all else fixed
    "ens_er_g": ("ens", "gauss_lab", "er", True),
    "ens_ecd_g": ("ens", "gauss_lab", "ecd", True),
    "dime_ens": ("ens", None, "learned", True),
    # DIME with 2x and 4x value-network training steps
    "dime_x2": ("sup", None, "learned", True),
    "dime_x4": ("sup", None, "learned", True),
    # diversity-weighted variants: score x residual-variance fraction under a
    # label-free Gaussian fitted on the pool, which down-weights redundant proxies
    "ours_div": ("ens", "mfa_pool", "eig_div", True),
    "ours_supcond_div": ("ens", "gauss_lab", "eig_div", True),
    "ours_frozen": ("ssl_frozen", "mfa_pool", "eig", True),
    "ssl_frozen_rand": ("ssl_frozen", None, "rand", True),
    "ssl_er": ("ssl", "mfa_pool", "er", True),
    "ssl_ecd": ("ssl", "mfa_pool", "ecd", True),
    "ssl_eig_supcond": ("ssl", "gauss_lab", "eig", True),
    "sup_eig_sslcond": ("sup", "mfa_pool", "eig", True),
    "ssl_eig_gausspool": ("ssl", "gauss_pool", "eig", True),
    "ours_costblind": ("ssl", "mfa_pool", "eig", False),
    "eig_sup_costblind": ("sup", "gauss_lab", "eig", False),
}


DIME_VALUE_KEY = {"dime": "dime_value", "dime_ens": "dime_value_ens",
                  "dime_x2": "dime_value_x2", "dime_x4": "dime_value_x4"}


class Run:
    def __init__(self, cfg: RunConfig, merge: bool = False, status_file: str | None = None):
        """merge=True adds the methods in cfg.acq.methods to an existing run directory: the split
        and labeled rows are identical (seeded), models are retrained, and the new methods' metrics
        are merged into summary.json. Methods already present are skipped."""
        self.cfg = cfg
        self.merge = merge
        # explicit filename takes precedence; falls back to the merge/non-merge convention
        # so old call sites and ad hoc scripts keep working
        self.status_file = status_file or ("status_add.json" if merge else "status.json")
        self.out = Path(cfg.out_root) / cfg.run_id
        self.out.mkdir(parents=True, exist_ok=True)
        save_config(cfg, self.out / ("config_added.yaml" if merge else "config.yaml"))
        self.log = RunLogger(cfg.log_root, cfg.run_id)
        self.models: dict[str, object] = {}
        self.info: dict[str, dict] = {}

    # ------------------------------------------------------------------ data
    def prepare_data(self) -> None:
        cfg = self.cfg
        rng = self.rng
        d = build_dataset(cfg.data, cfg.data_root)
        X, y = d["X"], d["y"]
        self.n_classes = int(y[y >= 0].max()) + 1
        cohort = d.get("cohort")
        pool_mask = test_mask = None
        if cfg.data.pretrain_cohorts is not None and cohort is not None:
            keep = np.isin(cohort, list(cfg.data.pretrain_cohorts) + list(cfg.data.eval_cohorts or []))
            pool_mask = keep
            test_mask = np.isin(cohort, cfg.data.eval_cohorts) if cfg.data.eval_cohorts else keep
        split = make_split(y, cfg.data.n_test, rng, X, pool_mask=pool_mask, test_mask=test_mask)
        if cfg.data.tune_mode:
            # hyper-parameter selection never touches the test rows: the last n_val rows of the
            # stratified pool ordering become the evaluation rows and leave the pool
            val = split.pool_order[-cfg.data.n_val:]
            split.pool_order = split.pool_order[:-cfg.data.n_val]
            split.pool = np.array([r for r in split.pool if r not in set(val)])
            split.test = val
            split.mean = X[split.pool].mean(axis=0)
            split.std = X[split.pool].std(axis=0) + 1e-6
        Xs = standardise(X, split)
        self.split = split
        self.X_pool, self.y_pool = Xs[split.pool], y[split.pool]
        self.X_test, self.y_test = Xs[split.test].copy(), y[split.test]
        if cfg.data.test_noise_sd > 0:
            self.X_test += rng.normal(scale=cfg.data.test_noise_sd, size=self.X_test.shape).astype(np.float32)
        # MCAR missingness in the pool: missing entries are set to the (standardised) mean 0
        self.valid_pool = np.ones_like(self.X_pool, dtype=bool)
        if cfg.data.pool_missing_rate > 0:
            self.valid_pool = rng.random(self.X_pool.shape) >= cfg.data.pool_missing_rate
            self.X_pool = np.where(self.valid_pool, self.X_pool, 0.0).astype(np.float32)
        # unlabeled rows used for pretraining (optionally cohort- or fraction-restricted)
        ssl_rows = np.arange(len(split.pool))
        if cfg.data.pretrain_cohorts is not None and cohort is not None:
            ssl_rows = np.where(np.isin(cohort[split.pool], cfg.data.pretrain_cohorts))[0]
        if cfg.train.ssl_pool_frac < 1.0:
            k = max(int(len(ssl_rows) * cfg.train.ssl_pool_frac), 32)
            ssl_rows = rng.choice(ssl_rows, size=k, replace=False)
        self.ssl_rows = ssl_rows
        # labeled subset: nested stratified prefix of the pool ordering (restricted to eval cohorts)
        order = split.pool_order
        if cfg.data.eval_cohorts is not None and cohort is not None:
            order = order[np.isin(cohort[order], cfg.data.eval_cohorts)]
        lab = order[: min(cfg.n_labels, len(order))]
        pos = {r: i for i, r in enumerate(split.pool)}
        self.lab_rows = np.array([pos[r] for r in lab])
        self.X_lab, self.y_lab = self.X_pool[self.lab_rows], self.y_pool[self.lab_rows]
        self.valid_lab = self.valid_pool[self.lab_rows]
        self.d = Xs.shape[1]
        self.group = d.get("group")
        self.causal = d.get("causal")
        self.log.event("data", dataset=cfg.data.name, n=len(X), d=self.d, n_classes=self.n_classes,
                       n_test=len(split.test), n_pool=len(split.pool), n_ssl=len(ssl_rows),
                       n_pool_unlabeled=int((self.y_pool < 0).sum()), n_labeled=len(self.lab_rows),
                       label_counts=np.bincount(self.y_lab, minlength=self.n_classes).tolist())

    # ---------------------------------------------------------------- models
    def get(self, key: str):
        if key in self.models:
            return self.models[key]
        cfg = self.cfg
        t0 = time.time()
        if key == "mfa_pool":
            model, info = train_mfa(self.X_pool[self.ssl_rows], self.valid_pool[self.ssl_rows],
                                    cfg.model, cfg.train, cfg.mask, self.rng, self.log,
                                    deadline_s=cfg.train.max_minutes * 60 * 0.5)
        elif key == "mfa_lab":
            model, info = train_mfa(self.X_lab, self.valid_lab, cfg.model, cfg.train, cfg.mask,
                                    self.rng, None, deadline_s=cfg.train.max_minutes * 60 * 0.25)
        elif key == "ssl":
            model, info = train_head(self.get("mfa_pool"), self.X_lab, self.y_lab, self.n_classes,
                                     cfg.model, cfg.train, cfg.mask, self.rng, self.log, self.valid_lab,
                                     X_unl=self.X_pool[self.ssl_rows], valid_unl=self.valid_pool[self.ssl_rows])
        elif key == "ssl_frozen":
            # ablation: linear head on the frozen pretrained encoder, no fine-tuning
            import dataclasses as _dc

            tcfg = _dc.replace(cfg.train, finetune=False)
            model, info = train_head(self.get("mfa_pool"), self.X_lab, self.y_lab, self.n_classes,
                                     cfg.model, tcfg, cfg.mask, self.rng, self.log, self.valid_lab)
        elif key == "sup":
            model, info = train_sup_classifier(self.X_lab, self.y_lab, self.n_classes, cfg.sup_model,
                                               cfg.train, cfg.mask, self.rng, self.log, self.valid_lab)
        elif key == "aug":
            model, info = train_sup_classifier(self.X_lab, self.y_lab, self.n_classes, cfg.sup_model,
                                               cfg.train, cfg.mask, self.rng, self.log, self.valid_lab,
                                               pre=self.get("mfa_pool"))
        elif key == "augens":
            model, info = EnsembleClassifier([self.get("aug"), self.get("ssl")], [0.5, 0.5]), {}
        elif key == "sup2":
            model, info = train_sup_classifier(self.X_lab, self.y_lab, self.n_classes, cfg.sup_model,
                                               cfg.train, cfg.mask, self.rng, None, self.valid_lab)
        elif key == "ens":
            w = self.select_mixture_weight()
            model = EnsembleClassifier([self.get("ssl"), self.get("sup")], [w, 1.0 - w])
            info = {"mixture_weight": w}
            if cfg.train.st_rounds > 0:
                model, st_info = self.self_train(model, w)
                info.update(st_info)
        elif key == "ens05":
            model, info = EnsembleClassifier([self.get("ssl"), self.get("sup")], [0.5, 0.5]), {"mixture_weight": 0.5}
        elif key == "sup_ens":
            model, info = EnsembleClassifier([self.get("sup"), self.get("sup2")]), {}
        elif key == "pvae_lab":
            model, info = train_partial_vae(self.X_lab, self.y_lab, np.ones(len(self.y_lab)),
                                            self.n_classes, cfg.baseline, cfg.train, cfg.mask,
                                            self.rng, self.log, self.valid_lab,
                                            deadline_s=cfg.train.max_minutes * 60 * 0.25)
        elif key == "pvae_pool":
            rows = self.ssl_rows
            Xp, yp = self.X_pool[rows], self.y_pool[rows]
            # only the rows that are part of the labeled set contribute the label term
            lab = np.zeros(len(rows), dtype=np.float32)
            pos = {r: i for i, r in enumerate(rows)}
            for r in self.lab_rows:
                if r in pos:
                    lab[pos[r]] = 1.0
            yp = np.where(lab > 0, yp, 0)
            model, info = train_partial_vae(Xp, yp, lab, self.n_classes, cfg.baseline, cfg.train,
                                            cfg.mask, self.rng, self.log, self.valid_pool[rows],
                                            deadline_s=cfg.train.max_minutes * 60 * 0.5)
            info["n_labeled_rows_in_pool"] = float(lab.sum())
        elif key in ("dime_value", "dime_value_ens", "dime_value_x2", "dime_value_x4"):
            import dataclasses as _dc

            predictor = self.get("ens" if key == "dime_value_ens" else "sup")
            mult = {"dime_value_x2": 2, "dime_value_x4": 4}.get(key, 1)
            bcfg = _dc.replace(cfg.baseline, dime_steps=cfg.baseline.dime_steps * mult)
            model, info = train_dime_value(predictor, self.X_lab, self.y_lab, bcfg,
                                           cfg.mask, self.rng, self.log, self.valid_lab)
            info["dime_steps"] = bcfg.dime_steps
        elif key == "static_scores":
            from sklearn.feature_selection import f_classif

            f, _ = f_classif(self.X_lab, self.y_lab)
            model = np.nan_to_num(f, nan=0.0, posinf=0.0)
            info = {"top5": np.argsort(-model)[:5].tolist()}
        elif key == "gauss_lab":
            # missing entries (if any) are already mean-imputed (0 after standardisation)
            model = GaussianConditional(self.X_lab)
            info = {"shrinkage": model.shrinkage}
        elif key == "gauss_pool":
            model = GaussianConditional(self.X_pool[self.ssl_rows])
            info = {"shrinkage": model.shrinkage}
        else:
            raise KeyError(key)
        info["seconds"] = time.time() - t0
        self.models[key] = model
        self.info[key] = info
        self.log.event("model", key=key, seconds=info["seconds"],
                       **{k: v for k, v in info.items() if k not in ("history", "seconds")})
        return model

    def select_mixture_weight(self) -> float:
        """Weight of the pretrained member, chosen by stratified k-fold cross-validation on the
        labeled rows: held-out log-likelihood under the training mask distribution, averaged
        over `mix_masks` random masks per row. Uses labels only; never touches test rows."""
        cfg = self.cfg
        if cfg.train.mix_folds <= 0:
            return float(cfg.train.mix_fixed)
        from sklearn.model_selection import StratifiedKFold

        from afa_ssl.data.masks import masks_tensor

        grid = [float(w) for w in cfg.train.mix_grid]
        ll = np.zeros(len(grid))
        n_splits = min(cfg.train.mix_folds, int(np.bincount(self.y_lab).min()), len(self.y_lab) // 2)
        if n_splits < 2:
            self.log.event("mixture", note="too few labels per class for CV; fixed weight", weight=cfg.train.mix_fixed)
            return float(cfg.train.mix_fixed)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
        t0 = time.time()
        for tr, te in skf.split(self.X_lab, self.y_lab):
            ssl_k, _ = train_head(self.get("mfa_pool"), self.X_lab[tr], self.y_lab[tr], self.n_classes, cfg.model,
                                  cfg.train, cfg.mask, self.rng, None, self.valid_lab[tr])
            sup_k, _ = train_sup_classifier(self.X_lab[tr], self.y_lab[tr], self.n_classes, cfg.sup_model,
                                            cfg.train, cfg.mask, self.rng, None, self.valid_lab[tr])
            xt, yt = torch.from_numpy(self.X_lab[te]), torch.from_numpy(self.y_lab[te])
            vt = torch.from_numpy(self.valid_lab[te].astype(np.float32))
            with torch.no_grad():
                for _ in range(cfg.train.mix_masks):
                    m = masks_tensor(len(te), self.d, cfg.mask.budget_max, cfg.mask.budget_mix, self.rng) * vt
                    p_ssl = torch.softmax(ssl_k(xt, m), -1)[torch.arange(len(te)), yt]
                    p_sup = torch.softmax(sup_k(xt, m), -1)[torch.arange(len(te)), yt]
                    for i, w in enumerate(grid):
                        ll[i] += torch.log((w * p_ssl + (1 - w) * p_sup).clamp_min(1e-12)).sum().item()
        best = int(np.argmax(ll))
        w = grid[best]
        self.log.event("mixture", weight=w, folds=n_splits, ll_per_row=(ll / (len(self.y_lab) * cfg.train.mix_masks)).round(4).tolist(),
                       seconds=time.time() - t0)
        self.info["mixture"] = {"weight": w, "ll": ll.tolist(), "grid": grid, "seconds": time.time() - t0}
        return w

    def self_train(self, model, w: float):
        """Semi-supervised rounds: pseudo-label pool rows whose full-view ensemble prediction is
        confident, retrain both members on labeled + pseudo-labeled rows (pseudo rows weighted by
        st_weight), and rebuild the ensemble. Labels of pool rows are never read."""
        cfg = self.cfg
        X_unl = self.X_pool[self.ssl_rows]
        V_unl = self.valid_pool[self.ssl_rows]
        info = {}
        for r in range(1, cfg.train.st_rounds + 1):
            xt = torch.from_numpy(X_unl)
            p = predict_proba(model, xt, torch.from_numpy(V_unl.astype(np.float32)))
            conf, pseudo = p.max(dim=-1)
            keep = (conf >= cfg.train.st_threshold).numpy()
            n_keep = int(keep.sum())
            X_aug = np.concatenate([self.X_lab, X_unl[keep]])
            y_aug = np.concatenate([self.y_lab, pseudo.numpy()[keep]])
            v_aug = np.concatenate([self.valid_lab, V_unl[keep]])
            wts = np.concatenate([np.ones(len(self.y_lab)), np.full(n_keep, cfg.train.st_weight)]).astype(np.float32)
            ssl_r, _ = train_head(self.get("mfa_pool"), X_aug, y_aug, self.n_classes, cfg.model, cfg.train,
                                  cfg.mask, self.rng, None, v_aug, sample_weight=wts)
            sup_r, _ = train_sup_classifier(X_aug, y_aug, self.n_classes, cfg.sup_model, cfg.train, cfg.mask,
                                            self.rng, None, v_aug, sample_weight=wts)
            model = EnsembleClassifier([ssl_r, sup_r], [w, 1.0 - w])
            self.models["ssl"], self.models["sup"] = ssl_r, sup_r
            # pseudo-label accuracy is a diagnostic only (uses hidden pool labels), never used for training
            true = self.y_pool[self.ssl_rows][keep]
            acc = float((pseudo.numpy()[keep] == true).mean()) if n_keep else float("nan")
            info[f"st_round{r}"] = {"n_pseudo": n_keep, "pseudo_acc_diag": acc}
            self.log.event("self_train", round=r, n_pseudo=n_keep, pseudo_acc_diag=acc)
        return model, info

    def conditional_for(self, key: str | None):
        if key is None:
            return None
        if key in ("mfa_pool", "mfa_lab"):
            return MFAConditional(self.get(key))
        return self.get(key)   # partial VAEs expose `conditional` themselves

    # ------------------------------------------------------------ evaluation
    def full_accuracy(self, clf) -> float:
        x = torch.from_numpy(self.X_test)
        p = predict_proba(clf, x, torch.ones_like(x))
        return float((p.argmax(-1).numpy() == self.y_test).mean())

    def run_method(self, name: str, costs: np.ndarray, cost_budget: float, max_steps: int
                   ) -> tuple[EpisodeResult, dict]:
        clf_key, cond_key, crit, cost_aware = METHODS[name]
        clf = self.get(clf_key)
        cond = self.conditional_for(cond_key)
        ep_rng = np.random.default_rng([self.cfg.seed, zlib.crc32(name.encode())])
        t0 = time.time()
        res = run_episode(clf, cond, crit, self.X_test, self.y_test, costs, cost_budget,
                          max_steps, ep_rng, cost_aware=cost_aware, gh_nodes=self.cfg.acq.gh_nodes,
                          chunk_rows=self.cfg.acq.chunk_rows,
                          static_scores=self.get("static_scores") if crit == "static" else None,
                          value_fn=(self.get(DIME_VALUE_KEY[name]) if crit == "learned" else None),
                          div_cond=self.get("gauss_pool") if crit.endswith("_div") else None)
        curve = accuracy_curve(res)
        metrics = {
            "method": name, "classifier": clf_key, "conditional": cond_key, "criterion": crit,
            "cost_aware": cost_aware, "seconds": time.time() - t0,
            "acc_curve": curve.tolist(), "acc_final": float(curve[-1]),
            "aucc": aucc_uniform(res) if self.cfg.acq.cost_model == "uniform" else aucc_cost(res, cost_budget),
            "aucc_uniform_steps": aucc_uniform(res),
            "acc_full": self.full_accuracy(clf),
            "mean_cost_final": float(res.cum_cost[:, -1].mean()),
            "mixture_weight": self.info.get("mixture", {}).get("weight") if clf_key == "ens" else None,
        }
        if self.causal is not None and self.group is not None:
            k = len(self.causal)
            metrics.update({
                "precision_at_k": precision_at_k(res, self.causal, k),
                "group_coverage_at_k": group_coverage(res, self.group, self.causal, k),
                "redundancy_at_k": redundancy_rate(res, self.group, k),
                "redundancy_all": redundancy_rate(res, self.group),
            })
        return res, metrics

    # ------------------------------------------------------------------ main
    def execute(self) -> dict:
        cfg = self.cfg
        t_start = time.time()
        self.rng = seed_everything(cfg.seed)
        self.log.event("start", run_id=cfg.run_id, config_hash=config_hash(cfg), git=git_commit(),
                       threads=torch.get_num_threads(), **to_dict(machine_state()))
        self.prepare_data()
        costs = make_costs(self.d, cfg.acq.cost_model, cfg.acq.cost_tiers, np.random.default_rng(cfg.seed))
        cost_budget = cfg.acq.cost_budget or float(cfg.acq.budget * costs.mean())
        max_steps = cfg.acq.budget if cfg.acq.cost_model == "uniform" else int(cfg.acq.budget * costs.max() / costs.min())
        np.save(self.out / "costs.npy", costs)
        summary: dict[str, dict] = {}
        previous = None
        if self.merge and (self.out / "summary.json").exists():
            previous = json.load(open(self.out / "summary.json"))
            summary = dict(previous["methods"])
        rows = []
        for name in cfg.acq.methods:
            if name not in METHODS:
                raise KeyError(f"unknown method {name}; known: {sorted(METHODS)}")
            if previous is not None and name in previous["methods"]:
                self.log.event("skip", method=name, note="already present")
                continue
            res, metrics = self.run_method(name, costs, cost_budget, max_steps)
            np.savez_compressed(self.out / f"{name}.npz", correct=res.correct, cum_cost=res.cum_cost,
                                acquired=res.acquired, prob_true=res.prob_true)
            summary[name] = metrics
            self.log.event("method", **{k: v for k, v in metrics.items() if k != "acc_curve"})
            rows.append({"run_id": cfg.run_id, "git_commit": git_commit(), "config_hash": config_hash(cfg),
                         "dataset": cfg.data.name, "method": name, "seed": cfg.seed,
                         "n_labels": len(self.lab_rows), "aucc": round(metrics["aucc"], 5),
                         "acc_final": round(metrics["acc_final"], 5), "acc_full": round(metrics["acc_full"], 5),
                         "cost_budget": cost_budget, "status": "done",
                         "runtime_s": round(metrics["seconds"], 1), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
        st = machine_state()
        out = {"run_id": cfg.run_id, "config_hash": config_hash(cfg), "git": git_commit(),
               "dataset": cfg.data.name, "seed": cfg.seed, "n_labels": int(len(self.lab_rows)),
               "n_test": int(len(self.y_test)), "d": self.d, "n_classes": self.n_classes,
               "cost_budget": cost_budget, "methods": summary,
               "model_info": {k: {kk: vv for kk, vv in v.items() if kk != "history"} for k, v in self.info.items()},
               "ssl_history": self.info.get("mfa_pool", {}).get("history"),
               "runtime_s": time.time() - t_start, "peak_rss_gb": st.rss_gb}
        if previous is not None:
            out["config_hash"] = previous["config_hash"]
            out["added_methods"] = sorted(set(summary) - set(previous["methods"]))
            out["model_info"] = {**previous.get("model_info", {}), **{f"added:{k}": v for k, v in out["model_info"].items()}}
            out["ssl_history"] = previous.get("ssl_history")
            out["runtime_s"] = previous.get("runtime_s", 0) + (time.time() - t_start)
        write_json(out, self.out / "summary.json")
        append_registry(Path(cfg.out_root) / "registry.csv", rows)
        write_json({"status": "done", "runtime_s": out["runtime_s"]},
                   self.out / self.status_file)
        self.log.event("done", runtime_s=out["runtime_s"], rss_gb=st.rss_gb)
        self.log.close()
        return out
