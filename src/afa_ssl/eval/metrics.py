"""Curve metrics computed from an EpisodeResult."""
from __future__ import annotations

import numpy as np

from afa_ssl.acquisition.policy import EpisodeResult


def accuracy_curve(res: EpisodeResult) -> np.ndarray:
    return res.correct.mean(axis=0)


def aucc_uniform(res: EpisodeResult) -> float:
    """Mean accuracy over budgets 1..T (normalised area under the cost-accuracy curve)."""
    return float(res.correct[:, 1:].mean())


def accuracy_at_cost_grid(res: EpisodeResult, grid: np.ndarray) -> np.ndarray:
    """For every grid point g, accuracy using each row's latest state with cum_cost <= g."""
    n, t1 = res.cum_cost.shape
    out = np.empty(len(grid))
    for k, g in enumerate(grid):
        ok = res.cum_cost <= g + 1e-6                        # (N, T+1), monotone along t
        last = ok.sum(axis=1) - 1                            # index of the last affordable state
        out[k] = res.correct[np.arange(n), last].mean()
    return out


def aucc_cost(res: EpisodeResult, cost_budget: float, n_grid: int = 50) -> float:
    grid = np.linspace(cost_budget / n_grid, cost_budget, n_grid)
    return float(accuracy_at_cost_grid(res, grid).mean())


def precision_at_k(res: EpisodeResult, causal: np.ndarray, k: int | None = None) -> float:
    """Fraction of the first k acquisitions that are in the causal set, averaged over rows."""
    k = k or len(causal)
    first = res.acquired[:, :k]
    hit = np.isin(first, causal) & (first >= 0)
    return float(hit.sum(axis=1).mean() / k)


def redundancy_rate(res: EpisodeResult, group: np.ndarray, k: int | None = None) -> float:
    """Fraction of the first k acquisitions whose correlation group already contained an
    acquired feature (a wasted acquisition if the group is redundant)."""
    n, t = res.acquired.shape
    k = k or t
    red = 0
    tot = 0
    for i in range(n):
        seen: set[int] = set()
        for f in res.acquired[i, :k]:
            if f < 0:
                continue
            g = int(group[f])
            red += g in seen
            seen.add(g)
            tot += 1
    return float(red / max(tot, 1))


def group_coverage(res: EpisodeResult, group: np.ndarray, causal: np.ndarray, k: int | None = None
                   ) -> float:
    """Fraction of causal groups touched within the first k acquisitions (proxies count)."""
    k = k or len(causal)
    cg = set(int(group[f]) for f in causal)
    cov = []
    for i in range(res.acquired.shape[0]):
        got = set(int(group[f]) for f in res.acquired[i, :k] if f >= 0)
        cov.append(len(got & cg) / len(cg))
    return float(np.mean(cov))
