import numpy as np
import torch

from afa_ssl.acquisition.criteria import (candidate_values, entropy, gauss_hermite, kl,
                                          scores_from_probs)


def test_entropy_basic():
    p = torch.tensor([[0.5, 0.5], [1.0, 0.0]])
    h = entropy(p)
    assert abs(h[0].item() - np.log(2)) < 1e-6
    assert abs(h[1].item()) < 1e-6


def test_gauss_hermite_moments():
    # E[x] = mu, E[x^2] = mu^2 + s^2, E[x^4] = mu^4 + 6 mu^2 s^2 + 3 s^4 exact with 3 nodes
    t, w = gauss_hermite(3)
    mu, s = 1.3, 0.7
    x = mu + s * t
    assert abs((w * x).sum() - mu) < 1e-5
    assert abs((w * x**2).sum() - (mu**2 + s**2)) < 1e-5
    assert abs((w * x**4).sum() - (mu**4 + 6 * mu**2 * s**2 + 3 * s**4)) < 1e-3
    assert abs(w.sum() - 1) < 1e-6


def test_candidate_values_shape():
    mu = torch.zeros(4, 7)
    sigma = torch.ones(4, 7)
    nodes = torch.tensor([-1.0, 0.0, 1.0])
    v = candidate_values(mu, sigma, nodes)
    assert v.shape == (4, 7, 3)
    assert torch.allclose(v[0, 0], nodes)


def _random_probs(n, d, j, c, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(n, d, j, c, generator=g)
    p = p / p.sum(-1, keepdim=True)
    w = torch.rand(j, generator=g)
    w = w / w.sum()
    p0 = torch.rand(n, c, generator=g)
    p0 = p0 / p0.sum(-1, keepdim=True)
    return p, w, p0


def test_eig_er_nonnegative_and_zero_when_uninformative():
    p, w, p0 = _random_probs(5, 6, 3, 4)
    eig = scores_from_probs(p, w, p0, "eig")
    er = scores_from_probs(p, w, p0, "er")
    assert (eig >= -1e-6).all()
    assert (er >= -1e-6).all()
    # identical node predictions -> no information gain, no error reduction
    same = p[:, :, :1, :].expand_as(p).contiguous()
    assert torch.allclose(scores_from_probs(same, w, p0, "eig"), torch.zeros(5, 6), atol=1e-6)
    assert torch.allclose(scores_from_probs(same, w, p0, "er"), torch.zeros(5, 6), atol=1e-6)


def test_ecd_equals_eig_plus_kl_pbar_p0():
    p, w, p0 = _random_probs(3, 4, 3, 5, seed=1)
    eig = scores_from_probs(p, w, p0, "eig")
    ecd = scores_from_probs(p, w, p0, "ecd")
    pbar = (w[None, None, :, None] * p).sum(2)
    assert torch.allclose(ecd, eig + kl(pbar, p0[:, None, :]), atol=1e-5)


def test_eig_upper_bound_by_log_classes():
    p, w, p0 = _random_probs(3, 4, 5, 3, seed=2)
    assert (scores_from_probs(p, w, p0, "eig") <= np.log(3) + 1e-6).all()
