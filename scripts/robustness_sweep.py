#!/usr/bin/env python
"""Robustness sweep on METABRIC at N_l=50, seeds 0-4: AUCC of the proposed method
against the better of the two supervised EIG criteria run in every condition
(eig_sup, eig_sup_ens), across test-time noise and pool missingness levels.

Writes experiments/aggregate/robustness_sweep.md and paper/tables/robustness_sweep.tex.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

EXP = Path("experiments")
SEEDS = range(5)
SUP = ("eig_sup", "eig_sup_ens")
CONDITIONS = [
    ("none", "main_metabric_s{s}_n50"),
    ("noise $\\sigma=0.25$", "noise_metabric_s{s}_n50_sd0.25"),
    ("noise $\\sigma=0.5$", "noise_metabric_s{s}_n50"),
    ("noise $\\sigma=1.0$", "noise_metabric_s{s}_n50_sd1.0"),
    ("missing 10\\%", "missing_metabric_s{s}_n50_m0.1"),
    ("missing 30\\%", "missing_metabric_s{s}_n50"),
    ("missing 50\\%", "missing_metabric_s{s}_n50_m0.5"),
    ("cross-cohort transfer", "transfer_metabric_s{s}_n50"),
]


def load(pattern: str):
    runs = []
    for s in SEEDS:
        f = EXP / pattern.format(s=s) / "summary.json"
        if not f.exists():
            return None
        runs.append({m: v["aucc"] for m, v in json.load(open(f))["methods"].items()})
    return runs


def main() -> None:
    md = ["| condition | ours | best sup. | diff | p | ours, Gaussian cond. | diff | p |", "|" + "---|" * 8]
    tex = ["\\begin{tabular}{lccc}", "\\toprule",
           "condition & best supervised & ours $-$ best sup.\\ ($p$) & SS-EIG $-$ best sup.\\ ($p$) \\\\", "\\midrule"]
    for name, pat in CONDITIONS:
        runs = load(pat)
        if runs is None:
            print(f"missing runs for {name} ({pat}); skipped")
            continue
        o = np.array([r["ours"] for r in runs])
        best = max(SUP, key=lambda m: np.mean([r[m] for r in runs]))
        b = np.array([r[best] for r in runs])
        p = stats.ttest_rel(o, b).pvalue
        if all("ours_supcond" in r for r in runs):
            v = np.array([r["ours_supcond"] for r in runs])
            pv = stats.ttest_rel(v, b).pvalue
            vtxt_md = f"{v.mean():.3f} | {v.mean() - b.mean():+.3f} | {pv:.3f}"
            vtxt = f"${v.mean() - b.mean():+.3f}$ ({pv:.3f})"
        else:
            vtxt_md, vtxt = "-- | -- | --", "--"
        md.append(f"| {name} | {o.mean():.3f} | {b.mean():.3f} ({best}) | {o.mean() - b.mean():+.3f} | {p:.3f} | "
                  f"{vtxt_md} |")
        tex.append(f"{name} & {b.mean():.3f} & {vtxt} & ${o.mean() - b.mean():+.3f}$ ({p:.3f}) \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    Path("experiments/aggregate/robustness_sweep.md").write_text("\n".join(md) + "\n")
    Path("paper/tables/robustness_sweep.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
