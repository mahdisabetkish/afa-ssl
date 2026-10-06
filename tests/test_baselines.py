import numpy as np
import torch
import torch.nn.functional as F

from afa_ssl.models.dime import ValueNetwork, cmi_targets, rollout_masks
from afa_ssl.models.eddi import PartialVAE
from afa_ssl.models.mfa import SupClassifier


def test_partial_vae_encoder_is_permutation_invariant_and_ignores_unobserved():
    torch.manual_seed(0)
    vae = PartialVAE(6, 3, emb=4, hidden=16, latent=8)
    x = torch.randn(2, 6)
    m = torch.tensor([[1.0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 1, 0]])
    c = vae.aggregate(x, m)
    x2 = x.clone()
    x2[m < 0.5] = 99.0                       # unobserved values must not matter
    assert torch.allclose(c, vae.aggregate(x2, m), atol=1e-5)
    # the aggregate is a sum over observed features, so it is order independent by construction
    terms = vae.feature_terms(x) * m.unsqueeze(-1)
    assert torch.allclose(c, terms.sum(1), atol=1e-6)


def test_partial_vae_candidates_match_naive():
    torch.manual_seed(1)
    n, d, j = 3, 5, 3
    vae = PartialVAE(d, 4, emb=4, hidden=16, latent=8, k_samples=2)
    vae.eval()
    x = torch.randn(n, d)
    m = (torch.rand(n, d) < 0.4).float()
    vals = torch.randn(n, d, j)
    with torch.no_grad():
        fast = vae.forward_candidates(x, m, vals)
        for i in range(n):
            for f in range(d):
                if m[i, f] > 0.5:
                    continue
                for k in range(j):
                    x2, m2 = x[i].clone(), m[i].clone()
                    x2[f], m2[f] = vals[i, f, k], 1.0
                    assert torch.allclose(fast[i, f, k], vae(x2[None], m2[None])[0], atol=1e-5)


def test_partial_vae_forward_returns_log_probabilities():
    torch.manual_seed(2)
    vae = PartialVAE(4, 3, emb=4, hidden=16, latent=8, k_samples=3)
    vae.eval()
    x, m = torch.randn(5, 4), torch.ones(5, 4)
    with torch.no_grad():
        logp = vae(x, m)
    assert torch.allclose(logp.exp().sum(-1), torch.ones(5), atol=1e-5)


def test_partial_vae_conditional_shapes_and_positive_sigma():
    torch.manual_seed(3)
    vae = PartialVAE(7, 2, emb=4, hidden=16, latent=8, k_samples=2)
    mu, sigma = vae.conditional(torch.randn(4, 7), torch.zeros(4, 7))
    assert mu.shape == (4, 7) and sigma.shape == (4, 7)
    assert (sigma > 0).all()


def test_partial_vae_learns_label_from_one_feature():
    """With y determined by feature 0, observing it should make the predictive confident,
    and observing only an uninformative feature should not. Eight features, so the ELBO is
    not in the degenerate regime where the latent collapses."""
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    n, d = 512, 8
    X = rng.normal(size=(n, d)).astype(np.float32)
    X = (X - X.mean(0)) / X.std(0)
    y = (X[:, 0] > 0).astype(np.int64)
    xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    vae = PartialVAE(d, 2, emb=4, hidden=32, latent=8)
    opt = torch.optim.Adam(vae.parameters(), lr=3e-3)
    ones, zeros = torch.ones(n), torch.zeros(n)
    for _ in range(600):
        m = torch.from_numpy((rng.random((n, d)) < 0.5).astype(np.float32))
        loss = vae.loss(xt, m, yt, zeros, ones)
        opt.zero_grad()
        loss.backward()
        opt.step()
    vae.eval()
    with torch.no_grad():
        m_inf = torch.zeros(n, d)
        m_inf[:, 0] = 1.0
        m_unin = torch.zeros(n, d)
        m_unin[:, 1] = 1.0
        acc_inf = (vae(xt, m_inf).argmax(-1) == yt).float().mean()
        acc_unin = (vae(xt, m_unin).argmax(-1) == yt).float().mean()
    assert acc_inf > 0.9
    assert acc_unin < 0.65


def test_cmi_targets_match_naive():
    torch.manual_seed(4)
    n, d, c = 4, 6, 3
    clf = SupClassifier(d, 16, 8, c)
    x = torch.randn(n, d)
    y = torch.randint(0, c, (n,))
    m = (torch.rand(n, d) < 0.5).float()
    fast = cmi_targets(clf, x, y, m)
    with torch.no_grad():
        ce0 = F.cross_entropy(clf(x, m), y, reduction="none")
        for i in range(n):
            for f in range(d):
                if m[i, f] > 0.5:
                    continue
                m2 = m[i].clone()
                m2[f] = 1.0
                ce = F.cross_entropy(clf(x[i][None], m2[None]), y[i][None])
                assert abs(fast[i, f].item() - (ce0[i] - ce).item()) < 1e-5


def test_rollout_masks_respect_budget_and_validity():
    rng = np.random.default_rng(0)
    torch.manual_seed(5)
    n, d = 8, 10
    net = ValueNetwork(d, 16)
    x = torch.randn(n, d)
    valid = torch.ones(n, d)
    valid[:, :3] = 0.0                      # these features can never be acquired
    m = rollout_masks(net, x, budget_max=4, rng=rng, valid=valid)
    assert (m.sum(1) <= 4).all()
    assert m[:, :3].sum() == 0
    assert set(torch.unique(m).tolist()) <= {0.0, 1.0}


def test_value_network_learns_a_constant_ranking():
    """If feature 0 always gives the largest reduction, the trained network ranks it first."""
    rng = np.random.default_rng(1)
    torch.manual_seed(6)
    n, d = 128, 6
    x = torch.from_numpy(rng.normal(size=(n, d)).astype(np.float32))
    target = torch.zeros(n, d)
    target[:, 0] = 1.0
    net = ValueNetwork(d, 32)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(300):
        m = torch.from_numpy((rng.random((n, d)) < 0.3).astype(np.float32))
        w = 1.0 - m
        pred = net(x, m)
        loss = (((pred - target) ** 2) * w).sum() / w.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        scores = net(x, torch.zeros(n, d))
    assert (scores.argmax(1) == 0).float().mean() > 0.9
