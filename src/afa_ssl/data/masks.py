"""Mask sampler shared by SSL pretraining, head training and the supervised baseline.

m[i, f] = 1 means feature f is observed for row i. The number of observed features
k is drawn from a mixture: with probability `budget_mix` from U{0..budget_max}
(the states an acquisition episode visits) and otherwise from U{0..D}."""
from __future__ import annotations

import numpy as np
import torch


def sample_masks(n: int, d: int, budget_max: int, budget_mix: float,
                 rng: np.random.Generator) -> np.ndarray:
    use_budget = rng.random(n) < budget_mix
    k = np.where(use_budget, rng.integers(0, min(budget_max, d) + 1, size=n),
                 rng.integers(0, d + 1, size=n))
    u = rng.random((n, d))
    ranks = np.argsort(np.argsort(u, axis=1), axis=1)
    return (ranks < k[:, None]).astype(np.float32)


def masks_tensor(n: int, d: int, budget_max: int, budget_mix: float,
                 rng: np.random.Generator) -> torch.Tensor:
    return torch.from_numpy(sample_masks(n, d, budget_max, budget_mix, rng))
