import numpy as np

from afa_ssl.acquisition.cost import cost_normalise, make_costs


def test_uniform_costs():
    c = make_costs(10, "uniform", [1, 2, 4], np.random.default_rng(0))
    assert c.shape == (10,) and (c == 1).all()


def test_tier_costs_balanced_and_seeded():
    c1 = make_costs(30, "tiers", [1.0, 2.0, 4.0], np.random.default_rng(3))
    c2 = make_costs(30, "tiers", [1.0, 2.0, 4.0], np.random.default_rng(3))
    assert np.array_equal(c1, c2)
    assert sorted(np.unique(c1).tolist()) == [1.0, 2.0, 4.0]
    assert np.bincount(c1.astype(int))[[1, 2, 4]].tolist() == [10, 10, 10]


def test_cost_normalise():
    s = np.array([[2.0, 2.0, 2.0]])
    c = np.array([1.0, 2.0, 4.0])
    out = cost_normalise(s, c)
    assert np.allclose(out, [[2.0, 1.0, 0.5]])
    assert np.allclose(cost_normalise(s, c, cost_aware=False), s)
