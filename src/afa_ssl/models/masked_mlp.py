"""MLP over the pair (x * m, m).

The first layer is split into W_x (acting on x * m) and W_m (acting on m) so
that "what if feature f were observed with value v" can be evaluated by adding
v * W_x[:, f] + W_m[:, f] to the first-layer pre-activation. Scoring every
candidate feature then costs O(H) per candidate instead of O(2 D H)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMLP(nn.Module):
    def __init__(self, d: int, hidden: int, n_layers: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.d = d
        self.wx = nn.Linear(d, hidden)
        self.wm = nn.Linear(d, hidden, bias=False)
        layers: list[nn.Module] = []
        for _ in range(max(n_layers - 1, 0)):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, out_dim)

    def first(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self.wx(x * m) + self.wm(m)

    def rest(self, h: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(F.relu(h)))

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self.rest(self.first(x, m))

    def forward_candidates(self, x: torch.Tensor, m: torch.Tensor, values: torch.Tensor
                           ) -> torch.Tensor:
        """values: (N, D, J) hypothetical values of each currently unobserved feature.
        Returns (N, D, J, out_dim). Entries for observed features are meaningless
        and must be ignored by the caller."""
        h0 = self.first(x, m)                                    # (N, H)
        wx = self.wx.weight.t()                                  # (D, H)
        wm = self.wm.weight.t()                                  # (D, H)
        h = h0[:, None, None, :] + values[..., None] * wx[None, :, None, :] + wm[None, :, None, :]
        n, d, j, hid = h.shape
        return self.rest(h.reshape(n * d * j, hid)).reshape(n, d, j, -1)
