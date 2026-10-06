#!/usr/bin/env python
"""Empirical companion to Proposition 3, prop:bottleneck (paper Section 3.8): for each label, the
fraction of its class separation that lies in the leading Z principal directions
of the (label-free) feature covariance,

    R2_Z = sum_c n_c sum_{k<=Z} (v_k^T d_c)^2 / lambda_k
           ----------------------------------------------
           sum_c n_c sum_{k}    (v_k^T d_c)^2 / lambda_k ,

where Sigma = sum_k lambda_k v_k v_k^T is a Ledoit-Wolf covariance of the
standardized features and d_c is class c's mean minus the overall mean. The
denominator is the between-class Mahalanobis scatter, so R2_Z is the share of
it that a Z-dimensional reconstruction bottleneck (which, in the linear case,
keeps exactly the top-Z principal subspace) can retain. Z = 64 matches the
encoder's latent size.

Also correlates R2_Z with the observed AUCC gap (ours - best supervised) at
N_l=50 across labels. This is post hoc and has seven points; it is reported as a
consistency check of the proposition, not a test of it.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import stats
from sklearn.covariance import LedoitWolf

LABELS = [
    ("metabric", "intrinsic subtype"), ("metabric_grade", "histological grade"),
    ("metabric_intclust", "integrative cluster"), ("metabric_er", "ER status"),
    ("metabric_her2", "HER2 status"), ("metabric_semisynth", "semi-synthetic (6 genes)"),
    ("synth_scm", "synthetic SCM"),
]
Z_VALUES = (16, 32, 64)


def alignment(ds: str, zs=Z_VALUES) -> dict:
    d = np.load(f"data/processed/{ds}/data.npz")
    X, y = d["X"].astype(np.float64), d["y"]
    keep = y >= 0
    X, y = X[keep], y[keep]
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    lam, V = np.linalg.eigh(LedoitWolf().fit(X).covariance_)
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    mu = X.mean(0)
    contrib = np.zeros(len(lam))
    for c in np.unique(y):
        n_c = (y == c).sum()
        delta = X[y == c].mean(0) - mu
        contrib += n_c * (V.T @ delta) ** 2 / lam
    total = contrib.sum()
    return {z: float(contrib[:z].sum() / total) for z in zs}


def gap_at_50(ds: str) -> tuple[float, float]:
    ours, sups = [], {m: [] for m in ("eig_sup", "er_sup", "ecd_sup", "eig_sup_ens")}
    for seed in range(20):
        try:
            s = json.load(open(f"experiments/main_{ds}_s{seed}_n50/summary.json"))
        except FileNotFoundError:
            continue
        ours.append(s["methods"]["ours"]["aucc"])
        for m in sups:
            if m in s["methods"]:
                sups[m].append(s["methods"][m]["aucc"])
    best = max((m for m in sups if sups[m]), key=lambda m: np.mean(sups[m]))
    o, b = np.array(ours), np.array(sups[best][: len(ours)])
    return float((o - b).mean()), float(stats.ttest_rel(o, b).pvalue)


def main() -> None:
    rows = []
    for ds, name in LABELS:
        r2 = alignment(ds)
        gap, p = gap_at_50(ds)
        rows.append((name, ds, r2, gap, p))
        print(f"{name:28s} R2_16={r2[16]:.3f} R2_32={r2[32]:.3f} R2_64={r2[64]:.3f}  "
              f"gap@50={gap:+.3f} (p={p:.3f})")
    for z in Z_VALUES:
        rho, p = stats.spearmanr([r[2][z] for r in rows], [r[3] for r in rows])
        print(f"Spearman(R2_{z}, gap) across {len(rows)} labels: rho={rho:.3f}, p={p:.3f}")

    lines = ["\\begin{tabular}{lccc}", "\\toprule",
             "label & $R^2_{64}$ & gap at $\\Nl=50$ & $p$ \\\\", "\\midrule"]
    for name, _, r2, gap, p in sorted(rows, key=lambda r: -r[2][64]):
        lines.append(f"{name} & {r2[64]:.3f} & ${gap:+.3f}$ & {p:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open("paper/tables/signal_alignment.tex", "w").write("\n".join(lines))
    json.dump([{"label": r[0], "dataset": r[1], "r2": r[2], "gap": r[3], "p": r[4]} for r in rows],
              open("experiments/aggregate/signal_alignment.json", "w"), indent=1)
    print("wrote paper/tables/signal_alignment.tex, experiments/aggregate/signal_alignment.json")


if __name__ == "__main__":
    main()
