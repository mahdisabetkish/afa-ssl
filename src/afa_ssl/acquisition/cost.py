"""Feature cost models and cost-aware scoring."""
from __future__ import annotations

import numpy as np


def make_costs(d: int, model: str, tiers: list[float], rng: np.random.Generator) -> np.ndarray:
    if model == "uniform":
        return np.ones(d, dtype=np.float32)
    if model == "tiers":
        # equal-size tiers assigned by a seeded permutation of the features
        idx = rng.permutation(d)
        c = np.empty(d, dtype=np.float32)
        for t, block in enumerate(np.array_split(idx, len(tiers))):
            c[block] = tiers[t]
        return c
    raise ValueError(f"unknown cost model {model}")


def cost_normalise(scores: np.ndarray, costs: np.ndarray, cost_aware: bool = True) -> np.ndarray:
    """Utility per unit cost. Non-cost-aware scoring ignores costs (ablation)."""
    return scores / costs[None, :] if cost_aware else scores
