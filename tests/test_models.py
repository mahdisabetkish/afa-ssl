import numpy as np
import torch

from afa_ssl.acquisition.criteria import gauss_hermite
from afa_ssl.models.conditional import GaussianConditional
from afa_ssl.models.masked_mlp import MaskedMLP
from afa_ssl.models.mfa import MFA


def test_forward_candidates_matches_naive():
    torch.manual_seed(0)
    n, d, j = 5, 8, 3
    net = MaskedMLP(d, 16, 2, 4)
    x = torch.randn(n, d)
    m = (torch.rand(n, d) < 0.4).float()
    vals = torch.randn(n, d, j)
    fast = net.forward_candidates(x, m, vals)
    for i in range(n):
        for f in range(d):
            if m[i, f] > 0.5:
                continue
            for k in range(j):
                x2 = x[i].clone()
                x2[f] = vals[i, f, k]
                m2 = m[i].clone()
                m2[f] = 1.0
                naive = net(x2[None], m2[None])[0]
                assert torch.allclose(fast[i, f, k], naive, atol=1e-5)


def test_mfa_loss_only_on_masked_and_valid():
    torch.manual_seed(0)
    mfa = MFA(6, 8, 4)
    x = torch.randn(3, 6)
    m = torch.ones(3, 6)  # everything observed -> no target, loss 0
    assert mfa.loss(x, m).item() == 0.0
    m[:, 0] = 0.0
    valid = torch.ones(3, 6)
    valid[:, 0] = 0.0  # the only masked feature is invalid -> loss 0
    assert mfa.loss(x, m, valid).item() == 0.0
    assert mfa.loss(x, m).item() != 0.0


def test_mfa_learns_linear_dependence():
    """A feature that is an exact linear function of another should get a small
    conditional sigma once trained; an independent one should keep sigma ~ 1."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n = 512
    a = rng.normal(size=n)
    x = np.stack([a, 2 * a, rng.normal(size=n)], 1).astype(np.float32)
    x = (x - x.mean(0)) / x.std(0)
    xt = torch.from_numpy(x)
    mfa = MFA(3, 32, 8)
    opt = torch.optim.Adam(mfa.parameters(), lr=3e-3)
    for _ in range(600):
        m = torch.tensor([[1.0, 0.0, 0.0]]).repeat(n, 1)
        loss = mfa.loss(xt, m)
        opt.zero_grad()
        loss.backward()
        opt.step()
    mfa.eval()
    with torch.no_grad():
        mu, sigma = mfa.conditional(xt, torch.tensor([[1.0, 0.0, 0.0]]).repeat(n, 1))
    assert sigma[:, 1].mean() < 0.3
    assert sigma[:, 2].mean() > 0.6
    assert torch.corrcoef(torch.stack([mu[:, 1], xt[:, 1]]))[0, 1] > 0.95


def test_gaussian_conditional_closed_form():
    rng = np.random.default_rng(0)
    n = 2000
    z = rng.normal(size=(n, 2))
    X = np.stack([z[:, 0], 0.8 * z[:, 0] + 0.6 * z[:, 1], z[:, 1]], 1).astype(np.float32)
    g = GaussianConditional(X)
    x = torch.from_numpy(X[:5])
    m = torch.tensor([[1.0, 0.0, 0.0]]).repeat(5, 1)
    mu, sigma = g.conditional(x, m)
    # x1 | x0 ~ N(0.8 x0 * ..., 0.6^2) approximately (LW shrinkage is small at n=2000)
    assert torch.allclose(mu[:, 1], 0.8 * x[:, 0], atol=0.1)
    assert abs(sigma[0, 1].item() - 0.6) < 0.1
    assert abs(sigma[0, 2].item() - 1.0) < 0.1
    # nothing observed -> marginal
    mu0, s0 = g.conditional(x, torch.zeros(5, 3))
    assert torch.allclose(mu0, torch.from_numpy(X.mean(0)), atol=1e-4)
    assert abs(s0[0, 0].item() - 1.0) < 0.1


def test_gh_nodes_symmetric():
    t, w = gauss_hermite(3)
    assert abs(t[1]) < 1e-7 and abs(t[0] + t[2]) < 1e-7


def test_residual_head_candidates_match_naive():
    from afa_ssl.models.mfa import SSLClassifier

    torch.manual_seed(1)
    n, d, j, c = 4, 7, 3, 3
    mfa = MFA(d, 16, 8)
    clf = SSLClassifier(mfa, c, head="residual")
    x = torch.randn(n, d)
    m = (torch.rand(n, d) < 0.4).float()
    vals = torch.randn(n, d, j)
    fast = clf.forward_candidates(x, m, vals)
    for i in range(n):
        for f in range(d):
            if m[i, f] > 0.5:
                continue
            for k in range(j):
                x2, m2 = x[i].clone(), m[i].clone()
                x2[f], m2[f] = vals[i, f, k], 1.0
                assert torch.allclose(fast[i, f, k], clf(x2[None], m2[None])[0], atol=1e-5)
    assert len(list(clf.head_parameters())) == 4


def test_ensemble_averages_probabilities_and_candidates():
    from afa_ssl.models.mfa import EnsembleClassifier, SupClassifier

    torch.manual_seed(2)
    a, b = SupClassifier(5, 8, 4, 3), SupClassifier(5, 8, 4, 3)
    ens = EnsembleClassifier([a, b])
    x, m = torch.randn(6, 5), (torch.rand(6, 5) < 0.5).float()
    pa, pb = torch.softmax(a(x, m), -1), torch.softmax(b(x, m), -1)
    assert torch.allclose(torch.softmax(ens(x, m), -1), 0.5 * (pa + pb), atol=1e-6)
    vals = torch.randn(6, 5, 3)
    ca, cb = torch.softmax(a.forward_candidates(x, m, vals), -1), torch.softmax(b.forward_candidates(x, m, vals), -1)
    assert torch.allclose(torch.softmax(ens.forward_candidates(x, m, vals), -1), 0.5 * (ca + cb), atol=1e-6)


def test_weighted_ensemble_skips_zero_weight_member():
    from afa_ssl.models.mfa import EnsembleClassifier, SupClassifier

    torch.manual_seed(3)
    a, b = SupClassifier(5, 8, 4, 3), SupClassifier(5, 8, 4, 3)
    x, m = torch.randn(4, 5), torch.ones(4, 5)
    e1 = EnsembleClassifier([a, b], [1.0, 0.0])
    assert torch.allclose(torch.softmax(e1(x, m), -1), torch.softmax(a(x, m), -1), atol=1e-6)
    e = EnsembleClassifier([a, b], [0.25, 0.75])
    pa, pb = torch.softmax(a(x, m), -1), torch.softmax(b(x, m), -1)
    assert torch.allclose(torch.softmax(e(x, m), -1), 0.25 * pa + 0.75 * pb, atol=1e-6)


def test_aug_classifier_candidates_match_naive_and_pre_frozen():
    from afa_ssl.models.mfa import AugClassifier

    torch.manual_seed(4)
    n, d, j, c = 3, 6, 3, 4
    mfa = MFA(d, 16, 8)
    clf = AugClassifier(mfa.encoder, d, 16, 8, c)
    assert all(not p.requires_grad for p in clf.pre.parameters())
    assert all(not nm.startswith("pre.") for nm, p in clf.named_parameters() if p.requires_grad)
    x = torch.randn(n, d)
    m = (torch.rand(n, d) < 0.4).float()
    vals = torch.randn(n, d, j)
    fast = clf.forward_candidates(x, m, vals)
    for i in range(n):
        for f in range(d):
            if m[i, f] > 0.5:
                continue
            for k in range(j):
                x2, m2 = x[i].clone(), m[i].clone()
                x2[f], m2[f] = vals[i, f, k], 1.0
                assert torch.allclose(fast[i, f, k], clf(x2[None], m2[None])[0], atol=1e-5)
