"""Masked-feature autoencoder (the self-supervised model).

Encoder: MaskedMLP -> latent z. Decoder: z -> (mu, logvar) for every feature.
Trained with the Gaussian negative log-likelihood of the masked-out features."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from afa_ssl.models.masked_mlp import MaskedMLP


class MFA(nn.Module):
    def __init__(self, d: int, hidden: int, latent: int, n_layers: int = 2,
                 dropout: float = 0.0, logvar_min: float = -6.0, logvar_max: float = 4.0,
                 objective: str = "gaussian_nll"):
        super().__init__()
        self.d = d
        self.objective = objective
        # homoscedastic per-feature log-variance, only used by the "mse" objective
        self.register_buffer("fixed_logvar", torch.zeros(d))
        self.encoder = MaskedMLP(d, hidden, n_layers, latent, dropout)
        self.decoder = nn.Sequential(nn.ReLU(), nn.Linear(latent, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 2 * d))
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

    def encode(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, m)

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.decoder(z)
        mu, logvar = out[:, : self.d], out[:, self.d:]
        if self.objective == "mse":
            logvar = self.fixed_logvar[None, :].expand_as(mu)
        return mu, logvar.clamp(self.logvar_min, self.logvar_max)

    def conditional(self, x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-feature Gaussian p(x_f | x_obs): returns (mu, sigma), each (N, D)."""
        mu, logvar = self.decode(self.encode(x, m))
        return mu, torch.exp(0.5 * logvar)

    def loss(self, x: torch.Tensor, m: torch.Tensor, valid: torch.Tensor | None = None
             ) -> torch.Tensor:
        """Mean Gaussian NLL over features that are masked out (m = 0) and valid
        (not missing in the raw data)."""
        mu, logvar = self.decode(self.encode(x, m))
        target = (1.0 - m) if valid is None else (1.0 - m) * valid
        if self.objective == "mse":
            nll = 0.5 * (x - mu) ** 2
        else:
            nll = 0.5 * (logvar + (x - mu) ** 2 * torch.exp(-logvar))
        return (nll * target).sum() / target.sum().clamp_min(1.0)

    @torch.no_grad()
    def fit_fixed_logvar(self, x: torch.Tensor, m: torch.Tensor, valid: torch.Tensor | None = None
                         ) -> None:
        """For the mse objective: per-feature residual variance on masked-out entries."""
        mu, _ = self.decode(self.encode(x, m))
        target = (1.0 - m) if valid is None else (1.0 - m) * valid
        var = ((x - mu) ** 2 * target).sum(0) / target.sum(0).clamp_min(1.0)
        self.fixed_logvar.copy_(torch.log(var.clamp_min(1e-4)))


class SSLClassifier(nn.Module):
    """Pretrained MFA encoder (frozen or fine-tuned) plus a small head.

    head = "linear": logits = W z.
    head = "mlp": one hidden layer on z.
    head = "residual": logits = W z + W_x (x * m) + W_m m, a linear head that also reads the raw
    masked features, so labels carried by a few specific features are not lost in the latent."""

    def __init__(self, mfa: MFA, n_classes: int, head: str = "linear", hidden: int = 64):
        super().__init__()
        self.encoder = mfa.encoder
        self.kind = head
        latent = mfa.encoder.out.out_features
        d = mfa.encoder.d
        self.raw_x = self.raw_m = None
        if head == "linear":
            self.head = nn.Linear(latent, n_classes)
        elif head == "residual":
            self.head = nn.Linear(latent, n_classes)
            self.raw_x = nn.Linear(d, n_classes, bias=False)
            self.raw_m = nn.Linear(d, n_classes, bias=False)
        else:
            self.head = nn.Sequential(nn.ReLU(), nn.Linear(latent, hidden), nn.ReLU(),
                                      nn.Linear(hidden, n_classes))

    def head_parameters(self):
        mods = [self.head] + [m for m in (self.raw_x, self.raw_m) if m is not None]
        for mod in mods:
            yield from mod.parameters()

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        out = self.head(self.encoder(x, m))
        if self.raw_x is not None:
            out = out + self.raw_x(x * m) + self.raw_m(m)
        return out

    def forward_candidates(self, x: torch.Tensor, m: torch.Tensor, values: torch.Tensor
                           ) -> torch.Tensor:
        out = self.head(self.encoder.forward_candidates(x, m, values))       # (N, D, J, C)
        if self.raw_x is not None:
            base = self.raw_x(x * m) + self.raw_m(m)                          # (N, C)
            wx = self.raw_x.weight.t()                                        # (D, C)
            wm = self.raw_m.weight.t()
            out = out + base[:, None, None, :] + values[..., None] * wx[None, :, None, :] + wm[None, :, None, :]
        return out


class SupClassifier(nn.Module):
    """Same architecture as SSLClassifier (MaskedMLP encoder + linear head) but
    trained end to end on the labeled rows only."""

    def __init__(self, d: int, hidden: int, latent: int, n_classes: int, n_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.encoder = MaskedMLP(d, hidden, n_layers, latent, dropout)
        self.head = nn.Linear(latent, n_classes)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x, m))

    def forward_candidates(self, x: torch.Tensor, m: torch.Tensor, values: torch.Tensor
                           ) -> torch.Tensor:
        return self.head(self.encoder.forward_candidates(x, m, values))


def predict_proba(model: nn.Module, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return F.softmax(model(x, m), dim=-1)


class EnsembleClassifier(nn.Module):
    """Weighted mixture of member predictive distributions, p = sum_k w_k p_k. Returns
    log-probabilities so that softmax(output) equals the mixture. Members with zero
    weight are skipped."""

    def __init__(self, members: list[nn.Module], weights: list[float] | None = None):
        super().__init__()
        self.members = nn.ModuleList(members)
        w = torch.tensor(weights if weights is not None else [1.0 / len(members)] * len(members))
        self.register_buffer("log_w", torch.log(w.clamp_min(1e-12)))
        self.active = [k for k in range(len(members)) if w[k] > 0]

    def _mix(self, logps: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack([lp + self.log_w[k] for k, lp in zip(self.active, logps)])
        return torch.logsumexp(stacked, dim=0)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self._mix([F.log_softmax(self.members[k](x, m), dim=-1) for k in self.active])

    def forward_candidates(self, x: torch.Tensor, m: torch.Tensor, values: torch.Tensor
                           ) -> torch.Tensor:
        return self._mix([F.log_softmax(self.members[k].forward_candidates(x, m, values), dim=-1)
                          for k in self.active])


class AugClassifier(nn.Module):
    """From-scratch masked MLP whose first layer also reads the frozen pretrained latent
    z(x_obs) = e_theta(x * m, m):  h = W_x (x*m) + W_m m + W_z z + b.  It can ignore z, so it is
    never structurally weaker than the supervised classifier; with few labels it can lean on the
    low-dimensional summary of what the unlabeled pool knows about the unobserved features."""

    def __init__(self, pre: MaskedMLP, d: int, hidden: int, latent: int, n_classes: int,
                 n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.pre = pre
        for p_ in self.pre.parameters():
            p_.requires_grad_(False)
        z_dim = pre.out.out_features
        self.wx = nn.Linear(d, hidden)
        self.wm = nn.Linear(d, hidden, bias=False)
        self.wz = nn.Linear(z_dim, hidden, bias=False)
        layers: list[nn.Module] = []
        for _ in range(max(n_layers - 1, 0)):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, latent)
        self.head = nn.Linear(latent, n_classes)

    def trainable_parameters(self):
        for name, p_ in self.named_parameters():
            if not name.startswith("pre."):
                yield p_

    def rest(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(self.out(self.body(F.relu(h))))

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.pre(x, m)
        return self.rest(self.wx(x * m) + self.wm(m) + self.wz(z))

    def forward_candidates(self, x: torch.Tensor, m: torch.Tensor, values: torch.Tensor
                           ) -> torch.Tensor:
        with torch.no_grad():
            z0 = self.pre(x, m)                                       # (N, Z)  (unused: h0 excludes z)
            zc = self.pre.forward_candidates(x, m, values)           # (N, D, J, Z)
        h0 = self.wx(x * m) + self.wm(m)                             # (N, H)
        wx, wm = self.wx.weight.t(), self.wm.weight.t()              # (D, H)
        h = (h0[:, None, None, :] + values[..., None] * wx[None, :, None, :] + wm[None, :, None, :]
             + self.wz(zc))
        n, d, j, hid = h.shape
        return self.rest(h.reshape(n * d * j, hid)).reshape(n, d, j, -1)
