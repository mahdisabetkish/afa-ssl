"""DIME (Gadgil, Covert and Lee, ICLR 2024): a value network that predicts, for every
unobserved feature, how much the predictor's loss would drop if that feature were
observed. Acquisition takes the largest predicted value per unit cost.

The training target is the realised cross-entropy reduction of the predictor when the
candidate is revealed at its true value, which is a one-sample estimate of the
conditional mutual information. Masks are generated on policy: a rollout of the current
greedy policy, as in the paper, rather than uniformly at random.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from afa_ssl.models.masked_mlp import MaskedMLP


class ValueNetwork(nn.Module):
    def __init__(self, d: int, hidden: int = 128, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.net = MaskedMLP(d, hidden, n_layers, d, dropout)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self.net(x, m)


@torch.no_grad()
def rollout_masks(value_net: ValueNetwork, x: torch.Tensor, budget_max: int,
                  rng: np.random.Generator, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Masks visited by the current greedy policy: each row is rolled forward a random
    number of steps in {0, ..., budget_max}."""
    n, d = x.shape
    m = torch.zeros(n, d)
    steps = torch.from_numpy(rng.integers(0, min(budget_max, d) + 1, size=n).astype(np.int64))
    tie = torch.from_numpy(rng.random((n, d)).astype(np.float32)) * 1e-6
    for t in range(int(steps.max().item()) if len(steps) else 0):
        scores = value_net(x, m) + tie
        if valid is not None:
            scores = torch.where(valid > 0.5, scores, torch.full_like(scores, -float("inf")))
        scores = torch.where(m < 0.5, scores, torch.full_like(scores, -float("inf")))
        pick = scores.argmax(dim=1)
        active = (steps > t) & torch.isfinite(scores.max(dim=1).values)
        rows = torch.where(active)[0]
        m[rows, pick[rows]] = 1.0
    return m


def cmi_targets(predictor: nn.Module, x: torch.Tensor, y: torch.Tensor, m: torch.Tensor
                ) -> torch.Tensor:
    """Cross-entropy reduction for every feature revealed at its true value: (N, D)."""
    with torch.no_grad():
        ce0 = F.cross_entropy(predictor(x, m), y, reduction="none")          # (N,)
        logits = predictor.forward_candidates(x, m, x.unsqueeze(-1))         # (N, D, 1, C)
        n, d, _, c = logits.shape
        ce_f = F.cross_entropy(logits.reshape(n * d, c), y.repeat_interleave(d),
                               reduction="none").reshape(n, d)
        return ce0[:, None] - ce_f
