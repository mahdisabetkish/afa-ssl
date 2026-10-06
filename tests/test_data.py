import numpy as np

from afa_ssl.data.masks import sample_masks
from afa_ssl.data.splits import make_split, stratified_order
from afa_ssl.data.synthetic import make_semisynth, make_synth_scm


def test_mask_sampler_counts_and_regimes():
    rng = np.random.default_rng(0)
    m = sample_masks(2000, 50, budget_max=10, budget_mix=1.0, rng=rng)
    k = m.sum(1)
    assert k.max() <= 10 and k.min() >= 0
    m2 = sample_masks(2000, 50, budget_max=10, budget_mix=0.0, rng=rng)
    assert m2.sum(1).max() > 10
    assert set(np.unique(m)) <= {0.0, 1.0}


def test_stratified_order_prefixes():
    rng = np.random.default_rng(0)
    y = np.array([0] * 80 + [1] * 20)
    o = stratified_order(y, rng)
    first20 = y[o[:20]]
    assert abs((first20 == 1).mean() - 0.2) <= 0.06
    assert sorted(o.tolist()) == list(range(100))


def test_split_disjoint_and_nested():
    rng = np.random.default_rng(1)
    y = np.repeat(np.arange(3), 100)
    X = rng.normal(size=(300, 5))
    sp = make_split(y, 60, rng, X)
    assert len(sp.test) == 60
    assert set(sp.test).isdisjoint(sp.pool)
    assert len(sp.test) + len(sp.pool) == 300
    l50, l100 = sp.labeled(50), sp.labeled(100)
    assert set(l50) <= set(l100)
    assert np.bincount(y[sp.test]).tolist() == [20, 20, 20]


def test_synth_scm_ground_truth():
    d = make_synth_scm(500, 4, 5, 2, 3, 0.85, (0.5, 0.85), seed=0)
    assert d["X"].shape == (500, 20) and d["y"].shape == (500,)
    assert len(d["causal"]) == 2
    assert len(set(d["group"][d["causal"]])) == 2  # distinct groups
    assert abs(d["bayes_acc"] - 0.85) < 0.02
    c = np.corrcoef(d["X"], rowvar=False)
    g = d["group"]
    within = np.mean([abs(c[i, j]) for i in range(20) for j in range(20) if i < j and g[i] == g[j]])
    across = np.mean([abs(c[i, j]) for i in range(20) for j in range(20) if i < j and g[i] != g[j]])
    assert within > 0.4 and across < 0.15


def test_semisynth_ground_truth():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 30)).astype(np.float32)
    d = make_semisynth(X, n_causal=3, n_classes=3, target_bayes_acc=0.85, n_clusters=10, seed=0)
    assert len(d["causal"]) == 3 and d["y"].shape == (300,)
    assert abs(d["bayes_acc"] - 0.85) < 0.03


def test_split_with_unlabeled_rows():
    rng = np.random.default_rng(2)
    y = np.array([0] * 60 + [1] * 60 + [-1] * 40)
    X = rng.normal(size=(160, 4))
    sp = make_split(y, 40, rng, X)
    assert (y[sp.test] >= 0).all() and len(sp.test) == 40
    assert (y[sp.pool_order] >= 0).all()
    assert len(sp.pool) == 120 and (y[sp.pool] < 0).sum() == 40   # unlabeled rows stay in the pool
    assert len(sp.pool_order) == 80
