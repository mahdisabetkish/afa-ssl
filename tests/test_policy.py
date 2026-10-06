import numpy as np
import torch

from afa_ssl.acquisition.policy import run_episode
from afa_ssl.eval.metrics import (accuracy_at_cost_grid, aucc_uniform, precision_at_k,
                                  redundancy_rate)
from afa_ssl.models.conditional import GaussianConditional
from afa_ssl.models.mfa import SupClassifier


class OracleClf(torch.nn.Module):
    """Predicts class = sign of feature 0 when observed, else uniform."""

    def forward(self, x, m):
        s = x[:, 0] * m[:, 0]
        return torch.stack([-3 * s, 3 * s], 1)

    def forward_candidates(self, x, m, values):
        n, d, j = values.shape
        out = torch.zeros(n, d, j, 2)
        # observing feature 0 with value v -> logits (-3v, 3v); other features -> current
        cur = self.forward(x, m)
        out[:] = cur[:, None, None, :]
        out[:, 0, :, 0] = -3 * values[:, 0, :]
        out[:, 0, :, 1] = 3 * values[:, 0, :]
        return out


def test_episode_eig_picks_informative_feature_first():
    rng = np.random.default_rng(0)
    n, d = 40, 6
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    cond = GaussianConditional(X)
    costs = np.ones(d, dtype=np.float32)
    res = run_episode(OracleClf(), cond, "eig", X, y, costs, 3.0, 3, rng)
    assert (res.acquired[:, 0] == 0).all()
    assert res.correct[:, 1].mean() > 0.95
    assert res.correct.shape == (n, 4)
    assert np.allclose(res.cum_cost[:, -1], 3.0)
    assert 0.0 <= aucc_uniform(res) <= 1.0


def test_episode_respects_cost_budget_and_random_is_seeded():
    rng = np.random.default_rng(0)
    n, d = 10, 5
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = np.zeros(n, dtype=np.int64)
    clf = SupClassifier(d, 8, 4, 2)
    costs = np.array([1, 1, 4, 4, 4], dtype=np.float32)
    r1 = run_episode(clf, None, "rand", X, y, costs, 2.0, 5, np.random.default_rng(3))
    r2 = run_episode(clf, None, "rand", X, y, costs, 2.0, 5, np.random.default_rng(3))
    assert np.array_equal(r1.acquired, r2.acquired)
    assert (r1.cum_cost <= 2.0 + 1e-6).all()
    assert set(np.unique(r1.acquired[:, :2])) <= {0, 1}
    assert (r1.acquired[:, 2:] == -1).all()


def test_ground_truth_metrics():
    class R:
        acquired = np.array([[0, 5, 1], [5, 0, 2]])
    group = np.array([0, 0, 1, 1, 2, 2])
    causal = np.array([0, 2])
    assert precision_at_k(R, causal, 2) == 0.5  # row0: {0,5}->1 hit, row1: {5,0}->1 hit
    assert redundancy_rate(R, group, 3) == (1 + 0) / 6  # row0 third pick (1) repeats group 0


def test_accuracy_at_cost_grid():
    class R:
        cum_cost = np.array([[0, 1, 3], [0, 2, 4]], dtype=float)
        correct = np.array([[False, True, True], [False, False, True]])
    acc = accuracy_at_cost_grid(R, np.array([0.5, 2.0, 4.0]))
    assert np.allclose(acc, [0.0, 0.5, 1.0])


def test_static_order_follows_scores_and_costs():
    rng = np.random.default_rng(0)
    n, d = 6, 5
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = np.zeros(n, dtype=np.int64)
    clf = SupClassifier(d, 8, 4, 2)
    static = np.array([5.0, 1.0, 4.0, 3.0, 2.0])
    r = run_episode(clf, None, "static", X, y, np.ones(d, dtype=np.float32), 3.0, 3,
                    np.random.default_rng(0), static_scores=static)
    assert (r.acquired == np.array([0, 2, 3])).all()
    costs = np.array([10.0, 1, 1, 1, 1], dtype=np.float32)  # feature 0 too expensive per unit
    r2 = run_episode(clf, None, "static", X, y, costs, 3.0, 3, np.random.default_rng(0), static_scores=static)
    assert (r2.acquired[:, 0] == 2).all()


class TwoSignalClf(torch.nn.Module):
    """Label depends equally on features 0 and 2; feature 1 is an exact copy of feature 0.
    Without a diversity factor, EIG may buy 0 then its copy 1; with it, 1 should lose to 2."""

    def forward(self, x, m):
        s = x[:, 0] * m[:, 0] + x[:, 1] * m[:, 1] * (1 - m[:, 0]) + x[:, 2] * m[:, 2]
        return torch.stack([-3 * s, 3 * s], 1)

    def forward_candidates(self, x, m, values):
        n, d, j = values.shape
        out = torch.zeros(n, d, j, 2)
        for f in range(d):
            for k in range(j):
                x2, m2 = x.clone(), m.clone()
                x2[:, f] = values[:, f, k]
                m2[:, f] = 1.0
                out[:, f, k] = self.forward(x2, m2)
        return out


class _IndependentCond:
    """A deliberately naive conditional that ignores correlations (as a poorly calibrated
    pretrained decoder might): every unobserved feature stays at its marginal."""

    def conditional(self, x, m):
        return torch.zeros_like(x), torch.ones_like(x)


def test_diversity_factor_is_neutral_before_any_observation_and_penalises_proxies():
    from afa_ssl.models.conditional import GaussianConditional

    rng = np.random.default_rng(0)
    n = 400
    a, c = rng.normal(size=n), rng.normal(size=n)
    X = np.stack([a, a + 1e-3 * rng.normal(size=n), c, rng.normal(size=n)], 1).astype(np.float32)
    X = (X - X.mean(0)) / X.std(0)
    y = ((X[:, 0] + X[:, 2]) > 0).astype(np.int64)
    pool_gauss = GaussianConditional(X)

    # factor with nothing observed is ~1 for every feature
    _, s0 = pool_gauss.conditional(torch.from_numpy(X[:1]), torch.zeros(1, 4))
    _, s_marg = pool_gauss.conditional(torch.from_numpy(X[:1]), torch.zeros(1, 4))
    assert torch.allclose((s0 ** 2) / (s_marg ** 2), torch.ones(1, 4), atol=1e-5)

    # after observing feature 0, its near-copy (feature 1) keeps little residual variance
    # (Ledoit-Wolf shrinkage keeps it from reaching zero) while an independent feature keeps
    # almost all of it: the proxy is down-weighted by an order of magnitude
    m0 = torch.zeros(1, 4)
    m0[0, 0] = 1.0
    _, s1 = pool_gauss.conditional(torch.from_numpy(X[:1]), m0)
    r_copy = (s1[0, 1] ** 2) / (s_marg[0, 1] ** 2)
    r_indep = (s1[0, 2] ** 2) / (s_marg[0, 2] ** 2)
    assert r_copy < 0.1
    assert r_indep > 0.9
    assert r_indep / r_copy > 10

    costs = np.ones(4, dtype=np.float32)
    plain = run_episode(TwoSignalClf(), _IndependentCond(), "eig", X[:60], y[:60], costs, 2.0, 2,
                        np.random.default_rng(1))
    div = run_episode(TwoSignalClf(), _IndependentCond(), "eig_div", X[:60], y[:60], costs, 2.0, 2,
                      np.random.default_rng(1), div_cond=pool_gauss)
    # the diversity-weighted policy essentially never follows feature 0 with its copy
    copy_after_orig = lambda r: np.mean((r.acquired[:, 0] == 0) & (r.acquired[:, 1] == 1))
    assert copy_after_orig(div) <= copy_after_orig(plain)
    assert copy_after_orig(div) < 0.05
