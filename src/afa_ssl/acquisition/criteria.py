"""Acquisition criteria evaluated from candidate predictive distributions.

Notation (per test row): p0 = current predictive p(y | x_obs); for each
unobserved feature f and quadrature node j, p_j = p(y | x_obs, x_f = v_j) with
weights w_j; pbar = sum_j w_j p_j is the predictive after marginalising x_f under
the conditional model.

  EIG_f = H[pbar] - sum_j w_j H[p_j]          (mutual information I(y; x_f | x_obs) under the model)
  ER_f  = sum_j w_j max_c p_j,c - max_c pbar_c (expected reduction of Bayes error)
  ECD_f = sum_j w_j KL(p_j || p0)              (expected divergence from the current prediction)

All entropies in nats. EIG and ER are non-negative by Jensen; ECD is EIG plus
KL(pbar || p0), which measures how inconsistent the classifier is with its own
marginalisation."""
from __future__ import annotations

import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss

EPS = 1e-12


def entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * torch.log(p.clamp_min(EPS))).sum(dim=-1)


def kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (p * (torch.log(p.clamp_min(EPS)) - torch.log(q.clamp_min(EPS)))).sum(dim=-1)


def gauss_hermite(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes t_j and weights w_j such that E_{x~N(mu,s^2)}[g(x)] ~= sum_j w_j g(mu + s t_j)."""
    t, w = hermgauss(n_nodes)
    return (np.sqrt(2.0) * t).astype(np.float32), (w / np.sqrt(np.pi)).astype(np.float32)


def candidate_values(mu: torch.Tensor, sigma: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
    """(N, D) conditional moments -> (N, D, J) quadrature points."""
    return mu[..., None] + sigma[..., None] * nodes[None, None, :]


def scores_from_probs(p_cand: torch.Tensor, weights: torch.Tensor, p0: torch.Tensor,
                      criterion: str) -> torch.Tensor:
    """p_cand: (N, D, J, C); weights: (J,); p0: (N, C). Returns (N, D)."""
    w = weights[None, None, :, None]
    pbar = (w * p_cand).sum(dim=2)                      # (N, D, C)
    if criterion == "eig":
        return entropy(pbar) - (weights[None, None, :] * entropy(p_cand)).sum(dim=2)
    if criterion == "er":
        return (weights[None, None, :] * p_cand.max(dim=-1).values).sum(dim=2) - pbar.max(dim=-1).values
    if criterion == "ecd":
        return (weights[None, None, :] * kl(p_cand, p0[:, None, None, :])).sum(dim=2)
    raise ValueError(f"unknown criterion {criterion}")
