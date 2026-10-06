#!/usr/bin/env python
"""Diversity factor: does multiplying the acquisition score by the
residual-variance fraction (ours_div, ours_supcond_div) change AUCC, redundancy
and recovery relative to the undiversified criterion? Seeds 0-4, paired tests.

Writes experiments/aggregate/diversity_check.md and paper/tables/diversity_check.tex.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

DATASETS = {"synth_scm": "synthetic SCM", "metabric_semisynth": "semi-synthetic",
            "metabric": "METABRIC", "tcga_brca": "TCGA"}
PAIRS = [("ours_div", "ours"), ("ours_supcond_div", "ours_supcond")]


def load(ds: str, n: int):
    out = []
    for s in range(5):
        f = Path("experiments") / f"main_{ds}_s{s}_n{n}" / "summary.json"
        if not f.exists():
            return None
        m = json.load(open(f))["methods"]
        if not all(a in m and b in m for a, b in PAIRS):
            return None
        out.append(m)
    return out


def main() -> None:
    md = ["| dataset | N_l | pair | AUCC div | AUCC base | diff | p | red@k div | red@k base | prec@k div | prec@k base |",
          "|" + "---|" * 11]
    tex = ["\\begin{tabular}{llcccc}", "\\toprule",
           "dataset & $N_l$ & score & $\\Delta$AUCC ($p$) & redundancy@$k$ & precision@$k$ \\\\", "\\midrule"]
    for ds, name in DATASETS.items():
        for n in (50, 200):
            runs = load(ds, n)
            if runs is None:
                print(f"incomplete: {ds} n={n}")
                continue
            for a, b in PAIRS:
                x = np.array([r[a]["aucc"] for r in runs]); y = np.array([r[b]["aucc"] for r in runs])
                p = stats.ttest_rel(x, y).pvalue
                rec = "precision_at_k" in runs[0][a]
                g = lambda m, k: np.mean([r[m][k] for r in runs]) if rec else float("nan")
                md.append(f"| {name} | {n} | {a} vs {b} | {x.mean():.3f} | {y.mean():.3f} | {x.mean()-y.mean():+.3f} | "
                          f"{p:.3f} | {g(a,'redundancy_at_k'):.3f} | {g(b,'redundancy_at_k'):.3f} | "
                          f"{g(a,'precision_at_k'):.3f} | {g(b,'precision_at_k'):.3f} |")
                lab = "SS-EIG" if b == "ours" else "ours"
                red = f"{g(b,'redundancy_at_k'):.3f} $\\to$ {g(a,'redundancy_at_k'):.3f}" if rec else "--"
                prec = f"{g(b,'precision_at_k'):.3f} $\\to$ {g(a,'precision_at_k'):.3f}" if rec else "--"
                tex.append(f"{name} & {n} & {lab} & ${x.mean()-y.mean():+.3f}$ ({p:.3f}) & {red} & {prec} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    Path("experiments/aggregate/diversity_check.md").write_text("\n".join(md) + "\n")
    Path("paper/tables/diversity_check.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
