"""Paired statistics across seeds."""
from __future__ import annotations

import numpy as np
from scipy import stats


def paired_summary(a: np.ndarray, b: np.ndarray) -> dict:
    """a, b: per-seed values of two methods (same seeds). Returns paired t-test,
    Wilcoxon (when n >= 5), Cohen's d_z, 95% CI of the mean difference and wins."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    out = {"n": n, "mean_a": a.mean(), "mean_b": b.mean(), "mean_diff": d.mean(),
           "std_diff": d.std(ddof=1) if n > 1 else 0.0, "wins": int((d > 0).sum())}
    if n > 1 and d.std(ddof=1) > 0:
        t, p = stats.ttest_rel(a, b)
        out["t"] = float(t)
        out["p_t"] = float(p)
        out["d_z"] = float(d.mean() / d.std(ddof=1))
        h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
        out["ci95"] = (float(d.mean() - h), float(d.mean() + h))
    else:
        out.update({"t": float("nan"), "p_t": float("nan"), "d_z": float("nan"),
                    "ci95": (float("nan"), float("nan"))})
    if n >= 5 and np.any(d != 0):
        try:
            out["p_wilcoxon"] = float(stats.wilcoxon(a, b).pvalue)
        except ValueError:
            out["p_wilcoxon"] = float("nan")
    else:
        out["p_wilcoxon"] = float("nan")
    return out
