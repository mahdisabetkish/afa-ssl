#!/usr/bin/env python
"""Apply the decision rule (fixed before any run) to the non-omics dataset, UCI HAR.

The claim "the advantage extends beyond gene expression" is made iff the proposed method's mean
AUCC exceeds the best supervised criterion (max over eig_sup, er_sup, ecd_sup, eig_sup_ens, chosen
per cell by mean, as in the primary analysis) in >= 3 of the 4 cells (N_l in {25, 50, 100, 200},
seeds 0-9) AND the one-sided t-test across seeds of the per-seed mean difference over the 4 cells
gives p < 0.05. Per-cell paired t-tests with BH and BY across the 4 cells; Fisher's combined test for
comparability only; secondary comparisons against SS-EIG and against the same predictor in random
order. Writes experiments/aggregate/har_check.{md,json} and paper/tables/har.tex.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

DATASET = "uci_har"
LEVELS = (25, 50, 100, 200)
SEEDS = range(10)
SUP = ["eig_sup", "er_sup", "ecd_sup", "eig_sup_ens"]
NAMES = {"eig_sup": "EIG", "er_sup": "ER", "ecd_sup": "ECD", "eig_sup_ens": "EIG (sup.\\ ensemble)"}


def load(n: int):
    runs = []
    for s in SEEDS:
        f = Path("experiments") / f"main_{DATASET}_s{s}_n{n}" / "summary.json"
        if not f.exists():
            return None
        runs.append({m: v["aucc"] for m, v in json.load(open(f))["methods"].items()})
    return runs


def paired(a: np.ndarray, b: np.ndarray):
    d = a - b
    p = stats.ttest_rel(a, b).pvalue
    return d.mean(), p, (p / 2 if d.mean() > 0 else 1 - p / 2)


def one_sided(x: np.ndarray) -> float:
    t = stats.ttest_1samp(x, 0.0)
    return t.pvalue / 2 if x.mean() > 0 else 1 - t.pvalue / 2


def main() -> None:
    data = {n: load(n) for n in LEVELS}
    missing = [n for n, r in data.items() if r is None]
    if missing:
        print(f"incomplete: N_l {missing} not finished for all 10 seeds")
        return
    rows, per_seed, per_seed_ss = [], [], []
    for n in LEVELS:
        R = data[n]
        best = max(SUP, key=lambda m: np.mean([r[m] for r in R]))
        o = np.array([r["ours_supcond"] for r in R])
        b = np.array([r[best] for r in R])
        ss = np.array([r["ours"] for r in R])
        er = np.array([r["ens_rand"] for r in R])
        per_seed.append(o - b)
        per_seed_ss.append(o - ss)
        d, p, p1 = paired(o, b)
        rows.append({"n": n, "best": best, "ours": o.mean(), "ours_sd": o.std(), "sup": b.mean(),
                     "sup_sd": b.std(), "diff": d, "p": p, "p_better": p1,
                     "vs_ss": paired(o, ss)[:2], "vs_rand": paired(o, er)[:2], "ss_vs_sup": paired(ss, b)[:2]})
    pv = [r["p"] for r in rows]
    q_bh = multipletests(pv, method="fdr_bh")[1]
    q_by = multipletests(pv, method="fdr_by")[1]
    n_pos = sum(r["diff"] > 0 for r in rows)
    p_seed = float(one_sided(np.mean(per_seed, axis=0)))
    p_seed_ss = float(one_sided(np.mean(per_seed_ss, axis=0)))
    chi2 = -2 * np.sum(np.log(np.clip([r["p_better"] for r in rows], 1e-300, 1)))
    fisher = float(stats.chi2.sf(chi2, df=2 * len(rows)))
    ok = n_pos >= 3 and p_seed < 0.05
    md = ["# Non-omics dataset UCI HAR", "",
          f"ours > best supervised in {n_pos}/4 cells; seed-level one-sided p={p_seed:.3g} -> "
          f"{'EXTENDS' if ok else 'does not meet the rule'} (Fisher p={fisher:.3g}, reference only); "
          f"ours vs SS-EIG seed-level p={p_seed_ss:.3g}", "",
          "| N_l | best sup. | ours | best sup. AUCC | diff | p | q (BH) | q (BY) | vs SS-EIG (p) | vs same predictor, random order (p) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    tex = ["\\begin{tabular}{rlccccc}", "\\toprule",
           "$\\Nl$ & best supervised & AUCC & ours & ours $-$ best sup.\\ ($p$) & ours $-$ SS-EIG ($p$) & "
           "SS-EIG $-$ best sup.\\ ($p$) \\\\", "\\midrule"]
    fmt_p = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
    for r, qb, qy in zip(rows, q_bh, q_by):
        md.append(f"| {r['n']} | {r['best']} | {r['ours']:.3f} | {r['sup']:.3f} | {r['diff']:+.3f} | {r['p']:.4f} | "
                  f"{qb:.4f} | {qy:.4f} | {r['vs_ss'][0]:+.3f} ({r['vs_ss'][1]:.3f}) | "
                  f"{r['vs_rand'][0]:+.3f} ({r['vs_rand'][1]:.2e}) |")
        dag = "$^{\\dagger}$" if qb < 0.05 and r["diff"] > 0 else ""
        tex.append(f"{r['n']} & {NAMES[r['best']]} & {r['sup']:.3f} $\\pm$ {r['sup_sd']:.3f} & "
                   f"{r['ours']:.3f} $\\pm$ {r['ours_sd']:.3f} & ${r['diff']:+.3f}$ ({fmt_p(r['p'])}){dag} & "
                   f"${r['vs_ss'][0]:+.3f}$ ({fmt_p(r['vs_ss'][1])}) & ${r['ss_vs_sup'][0]:+.3f}$ ({fmt_p(r['ss_vs_sup'][1])}) \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    Path("experiments/aggregate/har_check.md").write_text("\n".join(md) + "\n")
    Path("experiments/aggregate/har_check.json").write_text(json.dumps(
        {"extends": bool(ok), "cells_ahead": int(n_pos), "p_seed": p_seed, "p_seed_vs_ssEIG": p_seed_ss,
         "fisher_p": fisher,
         "cells": [{"n": r["n"], "best": r["best"], "diff": r["diff"], "p": r["p"], "q_bh": float(qb), "q_by": float(qy),
                    "vs_ss": r["vs_ss"], "vs_rand": r["vs_rand"]} for r, qb, qy in zip(rows, q_bh, q_by)],
         "significant_cells": [r["n"] for r, qb in zip(rows, q_bh) if qb < 0.05 and r["diff"] > 0],
         "significant_cells_by": [r["n"] for r, qy in zip(rows, q_by) if qy < 0.05 and r["diff"] > 0]}, indent=1))
    Path("paper/tables/har.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
