#!/usr/bin/env python
"""Apply the decision rule (fixed before any run) to the independent TCGA GBM+LGG cohort.

The headline result replicates iff the proposed method's mean AUCC exceeds the best supervised
criterion (max over eig_sup, er_sup, ecd_sup, eig_sup_ens, chosen per cell by mean, as in the
primary analysis) in >= 3 of the 4 cells (N_l in {25, 50, 100, 200}, seeds 0-9) AND Fisher's
combined one-sided test over the 4 cells gives p < 0.05. Per-cell paired t-tests with BH across
the 4 cells; secondary comparisons against SS-EIG and against the same predictor in random order.
Writes experiments/aggregate/gbmlgg_check.md and paper/tables/gbmlgg.tex.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

LEVELS = (25, 50, 100, 200)
SEEDS = range(10)
SUP = ["eig_sup", "er_sup", "ecd_sup", "eig_sup_ens"]
NAMES = {"eig_sup": "EIG", "er_sup": "ER", "ecd_sup": "ECD", "eig_sup_ens": "EIG (sup.\\ ensemble)"}


def load(n: int):
    runs = []
    for s in SEEDS:
        f = Path("experiments") / f"main_tcga_gbmlgg_s{s}_n{n}" / "summary.json"
        if not f.exists():
            return None
        runs.append({m: v["aucc"] for m, v in json.load(open(f))["methods"].items()})
    return runs


def paired(a: np.ndarray, b: np.ndarray):
    d = a - b
    p = stats.ttest_rel(a, b).pvalue
    return d.mean(), p, (p / 2 if d.mean() > 0 else 1 - p / 2)


def main() -> None:
    data = {n: load(n) for n in LEVELS}
    missing = [n for n, r in data.items() if r is None]
    if missing:
        print(f"incomplete: N_l {missing} not finished for all 10 seeds")
        return
    rows = []
    for n in LEVELS:
        R = data[n]
        best = max(SUP, key=lambda m: np.mean([r[m] for r in R]))
        o = np.array([r["ours_supcond"] for r in R])
        b = np.array([r[best] for r in R])
        ss = np.array([r["ours"] for r in R])
        er = np.array([r["ens_rand"] for r in R])
        d, p, p1 = paired(o, b)
        rows.append({"n": n, "best": best, "ours": o.mean(), "ours_sd": o.std(), "sup": b.mean(),
                     "sup_sd": b.std(), "diff": d, "p": p, "p_better": p1,
                     "vs_ss": paired(o, ss)[:2], "vs_rand": paired(o, er)[:2], "ss_vs_sup": paired(ss, b)[:2]})
    _, q, _, _ = multipletests([r["p"] for r in rows], method="fdr_bh")
    n_pos = sum(r["diff"] > 0 for r in rows)
    chi2 = -2 * np.sum(np.log(np.clip([r["p_better"] for r in rows], 1e-300, 1)))
    fisher = float(stats.chi2.sf(chi2, df=2 * len(rows)))
    ok = n_pos >= 3 and fisher < 0.05
    md = ["# Independent cohort TCGA GBM+LGG", "",
          f"ours > best supervised in {n_pos}/4 cells; Fisher chi2={chi2:.2f}, df=8, p={fisher:.3g} -> "
          f"{'REPLICATES' if ok else 'does not meet the rule'}", "",
          "| N_l | best sup. | ours | best sup. AUCC | diff | p | q (BH) | vs SS-EIG (p) | vs same predictor, random order (p) |",
          "|---|---|---|---|---|---|---|---|---|"]
    tex = ["\\begin{tabular}{rlccccc}", "\\toprule",
           "$\\Nl$ & best supervised & AUCC & ours & ours $-$ best sup.\\ ($p$) & ours $-$ SS-EIG ($p$) & "
           "SS-EIG $-$ best sup.\\ ($p$) \\\\", "\\midrule"]
    fmt_p = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
    for r, qq in zip(rows, q):
        md.append(f"| {r['n']} | {r['best']} | {r['ours']:.3f} | {r['sup']:.3f} | {r['diff']:+.3f} | {r['p']:.4f} | "
                  f"{qq:.4f} | {r['vs_ss'][0]:+.3f} ({r['vs_ss'][1]:.3f}) | {r['vs_rand'][0]:+.3f} ({r['vs_rand'][1]:.2e}) |")
        dag = "$^{\\dagger}$" if qq < 0.05 and r["diff"] > 0 else ""
        tex.append(f"{r['n']} & {NAMES[r['best']]} & {r['sup']:.3f} $\\pm$ {r['sup_sd']:.3f} & "
                   f"{r['ours']:.3f} $\\pm$ {r['ours_sd']:.3f} & ${r['diff']:+.3f}$ ({fmt_p(r['p'])}){dag} & "
                   f"${r['vs_ss'][0]:+.3f}$ ({fmt_p(r['vs_ss'][1])}) & ${r['ss_vs_sup'][0]:+.3f}$ ({fmt_p(r['ss_vs_sup'][1])}) \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    Path("experiments/aggregate/gbmlgg_check.md").write_text("\n".join(md) + "\n")
    Path("experiments/aggregate/gbmlgg_check.json").write_text(json.dumps(
        {"replicates": bool(ok), "fisher_p": fisher,
         "significant_cells": [r["n"] for r, qq in zip(rows, q) if qq < 0.05 and r["diff"] > 0]}, indent=1))
    Path("paper/tables/gbmlgg.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
