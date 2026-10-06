"""Greedy sequential acquisition, batched over test rows.

A method is (classifier, conditional, criterion, cost_aware). The loop records,
after every acquisition, the cumulative cost and whether the prediction is
correct, plus which feature was bought, so that every curve and every
ground-truth metric can be computed afterwards without re-running anything."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from afa_ssl.acquisition.criteria import candidate_values, gauss_hermite, scores_from_probs


@dataclass
class EpisodeResult:
    correct: np.ndarray      # (N, T+1) correctness after t acquisitions (t = 0 is no features)
    cum_cost: np.ndarray     # (N, T+1)
    acquired: np.ndarray     # (N, T) feature index bought at step t, -1 if none affordable
    prob_true: np.ndarray    # (N, T+1) probability assigned to the true class
    n_steps: int


def _predict(clf, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return F.softmax(clf(x, m), dim=-1)


def _score_batch(clf, cond, criterion: str, x: torch.Tensor, m: torch.Tensor,
                 nodes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Scores (N, D) for one chunk of rows; +inf-safe, observed features get -inf later."""
    mu, sigma = cond.conditional(x, m)
    if criterion == "unc":
        return sigma**2
    vals = candidate_values(mu, sigma, nodes)
    with torch.no_grad():
        logits = clf.forward_candidates(x, m, vals)          # (N, D, J, C)
        p_cand = F.softmax(logits, dim=-1)
        p0 = _predict(clf, x, m)
    return scores_from_probs(p_cand, weights, p0, criterion)


def run_episode(clf, cond, criterion: str, X: np.ndarray, y: np.ndarray, costs: np.ndarray,
                cost_budget: float, max_steps: int, rng: np.random.Generator,
                cost_aware: bool = True, gh_nodes: int = 3, chunk_rows: int = 64,
                static_scores: np.ndarray | None = None, value_fn=None,
                div_cond=None) -> EpisodeResult:
    """`div_cond`, when given, multiplies each candidate's score by the fraction of its
    variance left unexplained by the observed features under that (label-free) conditional,
    sigma^2(x_f | x_O) / sigma^2(x_f). This down-weights proxies of correlation groups that
    are already covered, with no tuned weight. Criteria ending in "_div" use it."""
    n, d = X.shape
    xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    m = torch.zeros(n, d)
    ct = torch.from_numpy(costs)
    nodes_np, w_np = gauss_hermite(gh_nodes)
    nodes, weights = torch.from_numpy(nodes_np), torch.from_numpy(w_np)

    correct = np.zeros((n, max_steps + 1), dtype=bool)
    prob_true = np.zeros((n, max_steps + 1), dtype=np.float32)
    cum_cost = np.zeros((n, max_steps + 1), dtype=np.float32)
    acquired = -np.ones((n, max_steps), dtype=np.int64)

    def record(t: int) -> None:
        p = _predict(clf, xt, m)
        correct[:, t] = (p.argmax(dim=-1) == yt).numpy()
        prob_true[:, t] = p[torch.arange(n), yt].numpy()

    record(0)
    spent = torch.zeros(n)
    rand_keys = torch.from_numpy(rng.random((n, d)).astype(np.float32))  # fixed random order for `rand`
    base_criterion = criterion[:-4] if criterion.endswith("_div") else criterion
    if criterion.endswith("_div"):
        if div_cond is None:
            raise ValueError(f"criterion {criterion} needs div_cond")
        _, sig_marg = div_cond.conditional(xt[:1], torch.zeros(1, d))
        var_marg = (sig_marg[0] ** 2).clamp_min(1e-8)
    for t in range(1, max_steps + 1):
        if criterion == "rand":
            scores = rand_keys.clone()
        elif criterion == "learned":
            # DIME: a network scores every candidate directly from the current partial record
            with torch.no_grad():
                scores = value_fn(xt, m).clone()
            if cost_aware:
                scores = scores / ct[None, :]
            scores = scores + 1e-9 * rand_keys
        elif criterion == "static":
            # non-adaptive: one global ranking (e.g. ANOVA F on the labeled rows), same for every row
            scores = torch.from_numpy(static_scores.astype(np.float32))[None, :].repeat(n, 1)
            if cost_aware:
                scores = scores / ct[None, :]
            scores = scores + 1e-9 * rand_keys
        else:
            scores = torch.empty(n, d)
            for s in range(0, n, chunk_rows):
                e = min(s + chunk_rows, n)
                scores[s:e] = _score_batch(clf, cond, base_criterion, xt[s:e], m[s:e], nodes, weights)
            if criterion.endswith("_div"):
                _, sig_cond = div_cond.conditional(xt, m)
                residual = ((sig_cond ** 2) / var_marg[None, :]).clamp(0.0, 1.0)
                scores = scores * residual
            if cost_aware:
                scores = scores / ct[None, :]
            # random tie-breaking on exact ties (e.g. all-zero scores)
            scores = scores + 1e-9 * rand_keys
        affordable = (ct[None, :] <= (cost_budget - spent)[:, None]) & (m < 0.5)
        scores = torch.where(affordable, scores, torch.full_like(scores, -float("inf")))
        best = scores.argmax(dim=1)
        can = affordable.any(dim=1)
        rows = torch.where(can)[0]
        m[rows, best[rows]] = 1.0
        spent[rows] += ct[best[rows]]
        acquired[rows.numpy(), t - 1] = best[rows].numpy()
        cum_cost[:, t] = spent.numpy()
        record(t)
        if not can.any():
            # nothing affordable for anyone: pad the remaining steps
            for tt in range(t + 1, max_steps + 1):
                correct[:, tt] = correct[:, t]
                prob_true[:, tt] = prob_true[:, t]
                cum_cost[:, tt] = cum_cost[:, t]
            break
    return EpisodeResult(correct=correct, cum_cost=cum_cost, acquired=acquired,
                         prob_true=prob_true, n_steps=max_steps)
