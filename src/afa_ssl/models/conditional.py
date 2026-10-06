"""Conditional models p(x_f | x_obs) used inside the acquisition criteria.

Two implementations share one interface: `conditional(x, m) -> (mu, sigma)`, each
of shape (N, D), meaningful only where m == 0.

- MFAConditional wraps the self-supervised decoder.
- GaussianConditional is the classical closed-form conditional of a multivariate
  Gaussian with a Ledoit-Wolf shrinkage covariance. It is the conditional model
  given to the supervised baselines (fitted on the labeled rows only) and, as an
  ablation, to the proposed method (fitted on the pool)."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from afa_ssl.models.mfa import MFA


class MFAConditional:
    def __init__(self, mfa: MFA):
        self.mfa = mfa

    def conditional(self, x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.mfa.conditional(x, m)


class GaussianConditional:
    def __init__(self, X: np.ndarray, min_var: float = 1e-4):
        self.mu = X.mean(axis=0).astype(np.float64)
        lw = LedoitWolf().fit(X.astype(np.float64))
        self.cov = lw.covariance_
        self.shrinkage = float(lw.shrinkage_)
        self.min_var = min_var

    def conditional(self, x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xn = x.detach().cpu().numpy().astype(np.float64)
        mn = m.detach().cpu().numpy() > 0.5
        n, d = xn.shape
        mu_out = np.tile(self.mu, (n, 1))
        var_out = np.tile(np.diag(self.cov), (n, 1))
        # rows sharing an observed set share the solve
        keys: dict[bytes, list[int]] = {}
        for i in range(n):
            keys.setdefault(mn[i].tobytes(), []).append(i)
        for key, rows in keys.items():
            obs = np.frombuffer(key, dtype=bool)
            if not obs.any():
                continue
            o = np.where(obs)[0]
            u = np.where(~obs)[0]
            if len(u) == 0:
                continue
            soo = self.cov[np.ix_(o, o)]
            suo = self.cov[np.ix_(u, o)]
            k = np.linalg.solve(soo, suo.T).T               # (|u|, |o|)
            delta = xn[np.ix_(rows, o)] - self.mu[o]        # (rows, |o|)
            mu_out[np.ix_(rows, u)] = self.mu[u] + delta @ k.T
            var_u = np.diag(self.cov)[u] - np.einsum("ij,ij->i", k, suo)
            var_out[np.ix_(rows, u)] = np.maximum(var_u, self.min_var)
        return (torch.from_numpy(mu_out.astype(np.float32)),
                torch.from_numpy(np.sqrt(var_out).astype(np.float32)))
