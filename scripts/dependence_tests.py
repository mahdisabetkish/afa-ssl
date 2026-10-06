#!/usr/bin/env python
"""Combined tests that remain valid when the cells of a family are dependent.

Cells of one cohort share test rows within a seed and use nested labeled sets, so their per-cell
p-values are positively dependent and Fisher's combination (which assumes independence) is
optimistic. Seeds are independent replicates, so for each family we (i) average the per-cell
differences within each seed and apply a one-sided one-sample t-test across seeds, and (ii) apply
Simes' test to the one-sided per-cell p-values, valid under positive (MTP2) dependence. Fisher's
p-value is listed for reference only. Writes experiments/aggregate/dependence_tests.json and
paper/tables/dependence_tests.tex.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

SUP = ["eig_sup", "er_sup", "ecd_sup", "eig_sup_ens"]


def get(ds: str, n: int, s: int):
    f = Path("experiments") / f"main_{ds}_s{s}_n{n}" / "summary.json"
    return {m: v["aucc"] for m, v in json.load(open(f))["methods"].items()}


def best_sup(ds: str, n: int, seeds) -> str:
    runs = [get(ds, n, s) for s in seeds if (Path("experiments") / f"main_{ds}_s{s}_n{n}" / "summary.json").exists()]
    return max(SUP, key=lambda m: np.mean([r[m] for r in runs if m in r]))


def one_sided(x: np.ndarray) -> float:
    t = stats.ttest_1samp(x, 0.0)
    return t.pvalue / 2 if x.mean() > 0 else 1 - t.pvalue / 2


def family(cells, seeds, diff) -> dict:
    d = np.array([[diff(get(ds, n, s), ds, n) for ds, n in cells] for s in seeds])
    per_cell = np.sort([one_sided(d[:, j]) for j in range(d.shape[1])])
    k = len(per_cell)
    simes = float(min(1.0, min(k * per_cell[i] / (i + 1) for i in range(k))))
    fisher = float(stats.chi2.sf(-2 * np.sum(np.log(np.clip(per_cell, 1e-300, 1))), 2 * k))
    return {"cells": k, "seeds": len(seeds), "mean_diff": float(d.mean()), "p_seed": float(one_sided(d.mean(1))),
            "p_simes": simes, "p_fisher": fisher}


def multiplicity_counts(prim, glioma, small) -> dict:
    """Cells significant after Benjamini-Hochberg and after Benjamini-Yekutieli (valid under any
    dependence), using the same two-sided per-cell paired p-values as the paper's BH analyses."""
    from statsmodels.stats.multitest import multipletests

    def count(p, d):
        bh, by = multipletests(p, method="fdr_bh")[0], multipletests(p, method="fdr_by")[0]
        return {"bh_ahead": int(sum(r and x > 0 for r, x in zip(bh, d))), "bh_behind": int(sum(r and x < 0 for r, x in zip(bh, d))),
                "by_ahead": int(sum(r and x > 0 for r, x in zip(by, d))), "by_behind": int(sum(r and x < 0 for r, x in zip(by, d)))}

    out = {}
    se = json.load(open("experiments/aggregate/stats_extra.json"))
    for m in ("ours_supcond", "ours"):
        out[f"primary {m}"] = count([c["p"] for c in se[m]["cells"]], [c["mean_diff"] for c in se[m]["cells"]])
    p, d = [], []
    for ds, n in glioma:
        runs = [get(ds, n, s) for s in range(10)]
        b = max(SUP, key=lambda m: np.mean([r[m] for r in runs]))
        x, y = np.array([r["ours_supcond"] for r in runs]), np.array([r[b] for r in runs])
        p.append(stats.ttest_rel(x, y).pvalue); d.append((x - y).mean())
    out["glioma"] = count(p, d)
    p, d = [], []
    for a, b in (("ens_er_g", "er_sup"), ("ens_ecd_g", "ecd_sup"), ("dime_ens", "dime")):
        for ds, n in small:
            runs = [get(ds, n, s) for s in range(5)]
            x, y = np.array([r[a] for r in runs]), np.array([r[b] for r in runs])
            p.append(stats.ttest_rel(x, y).pvalue); d.append((x - y).mean())
    out["drop-in (12)"] = count(p, d)
    return out


def main() -> None:
    prim = [("metabric", n) for n in (25, 50, 100, 200, 400, 100000)] + \
           [("tcga_brca", n) for n in (25, 50, 100, 200, 100000)]
    held = [(ds, n) for ds in ("metabric_er", "metabric_her2", "metabric_intclust", "metabric_grade",
                               "synth_scm", "metabric_semisynth") for n in (50, 200)]
    glioma = [("tcga_gbmlgg", n) for n in (25, 50, 100, 200)]
    har = [("uci_har", n) for n in (25, 50, 100, 200)]
    small = [("metabric", 25), ("metabric", 50), ("tcga_brca", 25), ("tcga_brca", 50)]
    bp = {c: best_sup(*c, range(20)) for c in prim}
    bg = {c: best_sup(*c, range(10)) for c in glioma}
    bh = {c: best_sup(*c, range(10)) for c in har}
    fams = [
        ("breast, proposed vs best supervised", "post-selection", prim, range(10),
         lambda r, ds, n: r["ours_supcond"] - r[bp[(ds, n)]]),
        ("breast, SS-EIG vs best supervised", "pre-registered", prim, range(10),
         lambda r, ds, n: r["ours"] - r[bp[(ds, n)]]),
        ("held-out, proposed vs SS-EIG", "rule before results", held, range(5),
         lambda r, ds, n: r["ours_supcond"] - r["ours"]),
        ("glioma, proposed vs best supervised", "rule first", glioma, range(10),
         lambda r, ds, n: r["ours_supcond"] - r[bg[(ds, n)]]),
        ("glioma, proposed vs SS-EIG", "pre-specified secondary", glioma, range(10),
         lambda r, ds, n: r["ours_supcond"] - r["ours"]),
        ("activity data, proposed vs best supervised", "rule first", har, range(10),
         lambda r, ds, n: r["ours_supcond"] - r[bh[(ds, n)]]),
        ("activity data, proposed vs SS-EIG", "pre-specified secondary", har, range(10),
         lambda r, ds, n: r["ours_supcond"] - r["ours"]),
        ("ER with vs without proposed predictor", "rule first", small, range(5), lambda r, ds, n: r["ens_er_g"] - r["er_sup"]),
        ("ECD with vs without proposed predictor", "rule first", small, range(5), lambda r, ds, n: r["ens_ecd_g"] - r["ecd_sup"]),
        ("DIME with vs without proposed predictor", "rule first", small, range(5), lambda r, ds, n: r["dime_ens"] - r["dime"]),
        ("proposed vs DIME with 4x training", "rule first", small, range(5), lambda r, ds, n: r["ours_supcond"] - r["dime_x4"]),
    ]
    res = {name: dict(status=st, **family(cells, seeds, f)) for name, st, cells, seeds, f in fams}
    res["_multiplicity_counts"] = multiplicity_counts(prim, glioma, small)
    Path("experiments/aggregate/dependence_tests.json").write_text(json.dumps(res, indent=1))
    fmt = lambda p: f"${p:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}$" if p < 1e-3 else f"{p:.3f}"
    lines = ["\\begin{tabular}{lccccc}", "\\toprule",
             "family & cells $\\times$ seeds & mean diff. & seed-level $p$ & Simes $p$ & Fisher $p$ \\\\", "\\midrule"]
    for name, r in res.items():
        if name.startswith("_"):
            continue
        lines.append(f"{name} & {r['cells']} $\\times$ {r['seeds']} & ${r['mean_diff']:+.4f}$ & {fmt(r['p_seed'])} & "
                     f"{fmt(r['p_simes'])} & {fmt(r['p_fisher'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    Path("paper/tables/dependence_tests.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps(res["_multiplicity_counts"]))
    for name, r in res.items():
        if name.startswith("_"):
            continue
        print(f"{name:42s} {r['cells']}x{r['seeds']:2d} {r['mean_diff']:+.4f} seed p={r['p_seed']:.2e} "
              f"Simes={r['p_simes']:.2e} Fisher={r['p_fisher']:.2e}")


if __name__ == "__main__":
    main()
