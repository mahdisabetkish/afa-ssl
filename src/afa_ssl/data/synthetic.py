"""Synthetic and semi-synthetic generators with known informative features."""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def random_nonlinear_logits(xs: np.ndarray, n_classes: int, rng: np.random.Generator) -> np.ndarray:
    """A random two-layer tanh network from the causal features to class logits."""
    k = xs.shape[1]
    w1 = rng.normal(size=(k, 4 * k)) / np.sqrt(k)
    b1 = rng.normal(size=4 * k) * 0.5
    w2 = rng.normal(size=(4 * k, n_classes)) / np.sqrt(4 * k)
    h = np.tanh(xs @ w1 + b1)
    logits = h @ w2
    # centre each class logit so that no class dominates by construction
    return logits - logits.mean(axis=0, keepdims=True)


def calibrate_temperature(logits: np.ndarray, target_acc: float, rng: np.random.Generator,
                          n_iter: int = 40) -> tuple[float, float]:
    """Bisection on the inverse temperature so that the Bayes accuracy (mean max
    class probability) matches `target_acc`."""
    lo, hi = 1e-3, 1e3
    for _ in range(n_iter):
        mid = np.sqrt(lo * hi)
        p = _softmax(logits * mid)
        acc = p.max(axis=1).mean()
        if acc < target_acc:
            lo = mid
        else:
            hi = mid
    beta = np.sqrt(lo * hi)
    return beta, _softmax(logits * beta).max(axis=1).mean()


def sample_labels(logits: np.ndarray, beta: float, rng: np.random.Generator) -> np.ndarray:
    p = _softmax(logits * beta)
    u = rng.random(len(p))
    return (p.cumsum(axis=1) < u[:, None]).sum(axis=1).astype(np.int64)


def make_synth_scm(n_samples: int, n_groups: int, group_size: int, n_causal: int,
                   n_classes: int, target_bayes_acc: float, within_corr: tuple[float, float],
                   seed: int) -> dict:
    """Latent-factor SCM. Feature group g loads on factor z_g; causal features come
    from distinct groups so every causal feature has group_size-1 redundant proxies."""
    rng = np.random.default_rng(seed)
    d = n_groups * group_size
    z = rng.normal(size=(n_samples, n_groups))
    x = np.empty((n_samples, d))
    group = np.repeat(np.arange(n_groups), group_size)
    for g in range(n_groups):
        # correlation between two features in the group is a^2 / (a^2 + b^2)
        rho = rng.uniform(*within_corr, size=group_size)
        a = np.sqrt(rho)
        b = np.sqrt(1.0 - rho)
        cols = np.where(group == g)[0]
        x[:, cols] = z[:, [g]] * a[None, :] + rng.normal(size=(n_samples, group_size)) * b[None, :]
    causal_groups = rng.choice(n_groups, size=n_causal, replace=False)
    causal = np.array([rng.choice(np.where(group == g)[0]) for g in causal_groups])
    logits = random_nonlinear_logits(x[:, causal], n_classes, rng)
    beta, bayes = calibrate_temperature(logits, target_bayes_acc, rng)
    y = sample_labels(logits, beta, rng)
    return {
        "X": x.astype(np.float32), "y": y, "feature_names": [f"f{i}" for i in range(d)],
        "class_names": [f"c{i}" for i in range(n_classes)], "group": group,
        "causal": causal, "bayes_acc": float(bayes), "beta": float(beta),
    }


def correlation_groups(X: np.ndarray, n_clusters: int, seed: int = 0) -> np.ndarray:
    """Average-linkage clustering of features on 1 - |corr|."""
    c = np.corrcoef(X, rowvar=False)
    c = np.nan_to_num(c, nan=0.0)
    dist = 1.0 - np.abs(c)
    iu = np.triu_indices_from(dist, k=1)
    z = linkage(dist[iu], method="average")
    return fcluster(z, t=n_clusters, criterion="maxclust") - 1


def make_semisynth(X: np.ndarray, n_causal: int, n_classes: int, target_bayes_acc: float,
                   n_clusters: int, seed: int) -> dict:
    """Real X, simulated y from `n_causal` standardised features drawn from distinct
    correlation clusters."""
    rng = np.random.default_rng(seed)
    group = correlation_groups(X, n_clusters, seed)
    sizes = np.bincount(group)
    eligible = np.where(sizes >= 2)[0]
    chosen_groups = rng.choice(eligible, size=n_causal, replace=False)
    causal = np.array([rng.choice(np.where(group == g)[0]) for g in chosen_groups])
    xs = X[:, causal]
    xs = (xs - xs.mean(0)) / (xs.std(0) + 1e-8)
    logits = random_nonlinear_logits(xs, n_classes, rng)
    beta, bayes = calibrate_temperature(logits, target_bayes_acc, rng)
    y = sample_labels(logits, beta, rng)
    return {"y": y, "group": group, "causal": causal, "bayes_acc": float(bayes),
            "beta": float(beta), "class_names": [f"c{i}" for i in range(n_classes)]}
