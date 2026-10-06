"""EDDI (Ma et al., ICML 2019): a partial VAE over features and label, used as an
acquisition model.

The encoder is permutation invariant over the observed set. Feature f with value v
is mapped to s_f(v) = h([v * b_f ; e_f]) with a learned embedding e_f and scale b_f,
and the observed set is summarised by c = sum_{f in O} s_f(x_f). Because the
aggregation is a sum, adding one candidate feature costs one extra term, which gives
the same cheap candidate scoring we use elsewhere.

The decoder maps the latent to a Gaussian per feature and a categorical over the
label, so one model provides both p(x_f | x_O) and p(y | x_O).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialVAE(nn.Module):
    def __init__(self, d: int, n_classes: int, emb: int = 16, hidden: int = 128,
                 latent: int = 32, k_samples: int = 1, logvar_min: float = -6.0,
                 logvar_max: float = 4.0, beta: float = 1.0, label_weight: float | None = None):
        super().__init__()
        self.d = d
        self.n_classes = n_classes
        self.latent = latent
        self.k_samples = k_samples
        self.beta = beta
        # EDDI treats the label as one more dimension of the data. With D = 200 features its term
        # would be swamped in the ELBO, so the weight is a tuned knob; `None` means D (the label
        # counts as much as all features together), 1.0 is the literal reading of the paper.
        self.label_weight = float(d if label_weight is None else label_weight)
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.feat_emb = nn.Parameter(torch.randn(d, emb) * 0.1)
        self.feat_scale = nn.Parameter(torch.ones(d))
        self.h = nn.Sequential(nn.Linear(emb + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.label_emb = nn.Parameter(torch.randn(n_classes, emb) * 0.1)
        self.h_label = nn.Sequential(nn.Linear(emb, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.to_z = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 2 * latent))
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                 nn.ReLU())
        self.dec_x = nn.Linear(hidden, 2 * d)
        self.dec_y = nn.Linear(hidden, n_classes)
        # fixed standard-normal draws, shared by every row and candidate (common random numbers,
        # so the acquisition scores are deterministic given the model)
        self.register_buffer("eps", torch.randn(k_samples, latent))

    # ------------------------------------------------------------------ encoder
    def feature_terms(self, x: torch.Tensor) -> torch.Tensor:
        """s_f(x_f) for every feature: (N, D, hidden)."""
        v = (x * self.feat_scale).unsqueeze(-1)                       # (N, D, 1)
        e = self.feat_emb.unsqueeze(0).expand(x.shape[0], -1, -1)     # (N, D, emb)
        return self.h(torch.cat([v, e], dim=-1))

    def aggregate(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """c = sum over observed features: (N, hidden)."""
        return (self.feature_terms(x) * m.unsqueeze(-1)).sum(dim=1)

    def label_term(self, y: torch.Tensor) -> torch.Tensor:
        return self.h_label(self.label_emb[y])

    def posterior(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.to_z(c)
        mu, logvar = out[..., : self.latent], out[..., self.latent:]
        return mu, logvar.clamp(self.logvar_min, self.logvar_max)

    # ------------------------------------------------------------------ decoder
    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.dec(z)
        xout = self.dec_x(h)
        mu, logvar = xout[..., : self.d], xout[..., self.d:]
        return mu, logvar.clamp(self.logvar_min, self.logvar_max), self.dec_y(h)

    def _z_samples(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """(K, ..., latent): posterior mean plus the fixed draws."""
        if self.k_samples <= 1:
            return mu.unsqueeze(0)
        sd = torch.exp(0.5 * logvar)
        return mu.unsqueeze(0) + sd.unsqueeze(0) * self.eps.view(
            self.k_samples, *([1] * (mu.dim() - 1)), self.latent)

    # ------------------------------------------------------- inference interface
    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Label logits from the observed features: log of the K-sample mixture."""
        mu, logvar = self.posterior(self.aggregate(x, m))
        z = self._z_samples(mu, logvar)                                # (K, N, latent)
        _, _, ylogits = self.decode(z)
        return torch.logsumexp(F.log_softmax(ylogits, dim=-1), dim=0) - math.log(self.k_samples)

    def forward_candidates(self, x: torch.Tensor, m: torch.Tensor, values: torch.Tensor
                           ) -> torch.Tensor:
        """values: (N, D, J) hypothetical values of each unobserved feature.
        Returns label logits (N, D, J, C). Observed features are meaningless here."""
        n, d, j = values.shape
        c0 = self.aggregate(x, m)                                      # (N, hidden)
        v = (values * self.feat_scale[None, :, None]).unsqueeze(-1)    # (N, D, J, 1)
        e = self.feat_emb[None, :, None, :].expand(n, d, j, -1)
        s = self.h(torch.cat([v, e], dim=-1))                          # (N, D, J, hidden)
        c = c0[:, None, None, :] + s
        mu, logvar = self.posterior(c)
        z = self._z_samples(mu, logvar)                                # (K, N, D, J, latent)
        _, _, ylogits = self.decode(z)
        return torch.logsumexp(F.log_softmax(ylogits, dim=-1), dim=0) - math.log(self.k_samples)

    def conditional(self, x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """p(x_f | x_O) as a Gaussian per feature, moment-matched over the latent samples."""
        with torch.no_grad():
            mu_z, logvar_z = self.posterior(self.aggregate(x, m))
            z = self._z_samples(mu_z, logvar_z)                        # (K, N, latent)
            mu_x, logvar_x, _ = self.decode(z)                         # (K, N, D)
            mean = mu_x.mean(dim=0)
            var = (torch.exp(logvar_x) + mu_x**2).mean(dim=0) - mean**2
            return mean, var.clamp_min(1e-6).sqrt()

    # ------------------------------------------------------------------ training
    def loss(self, x: torch.Tensor, m: torch.Tensor, y: torch.Tensor | None = None,
             y_observed: torch.Tensor | None = None, has_label: torch.Tensor | None = None,
             valid: torch.Tensor | None = None) -> torch.Tensor:
        """Negative ELBO. The encoder sees the observed features and, for rows where
        `y_observed` is 1, the label as well; the decoder reconstructs every feature and,
        for labeled rows, the label."""
        c = self.aggregate(x, m)
        if y is not None and y_observed is not None:
            c = c + self.label_term(y.clamp_min(0)) * y_observed.unsqueeze(-1)
        mu_z, logvar_z = self.posterior(c)
        std = torch.exp(0.5 * logvar_z)
        z = mu_z + std * torch.randn_like(std)
        mu_x, logvar_x, ylogits = self.decode(z)
        w = torch.ones_like(x) if valid is None else valid
        recon_x = (0.5 * (logvar_x + (x - mu_x) ** 2 * torch.exp(-logvar_x)) * w).sum(1)
        kl = 0.5 * (torch.exp(logvar_z) + mu_z**2 - 1.0 - logvar_z).sum(1)
        loss = recon_x + self.beta * kl
        if y is not None and has_label is not None and has_label.any():
            # predict the label only where it is known and was hidden from the encoder
            target = has_label * (1.0 - (y_observed if y_observed is not None else 0.0))
            if target.sum() > 0:
                ce = F.cross_entropy(ylogits, y.clamp_min(0), reduction="none")
                loss = loss + self.label_weight * ce * target
        return loss.mean()
