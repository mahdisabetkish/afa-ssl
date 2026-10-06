#!/usr/bin/env python
"""Apply the decision rule (fixed before any run): does the proposed method's advantage over DIME at
25 and 50 labels hold when DIME's value network gets 2x and 4x its training steps?

The claim is made iff ours_supcond - dime_x4 > 0 in >= 3 of the 4 cells (METABRIC and TCGA breast,
N_l in {25, 50}, seeds 0-4) AND Fisher's combined one-sided test gives p < 0.05. Also reported:
ours - dime (1x), ours - dime_x2, and DIME's own change from extra compute (dime_x4 - dime).
Writes experiments/aggregate/dime_budget_check.md and paper/tables/dime_budget.tex.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

CELLS = [("metabric", 25), ("metabric", 50), ("tcga_brca", 25), ("tcga_brca", 50)]
SEEDS = range(5)


def load(ds: str, n: int):
    out = []
    for s in SEEDS:
        m = json.load(open(Path("experiments") / f"main_{ds}_s{s}_n{n}" / "summary.json"))["methods"]
        if not all(k in m for k in ("ours_supcond", "dime", "dime_x2", "dime_x4")):
            return None
        out.append({k: m[k]["aucc"] for k in ("ours_supcond", "dime", "dime_x2", "dime_x4")})
    return out


def paired(a, b):
    d = a - b
    p = stats.ttest_rel(a, b).pvalue
    return d.mean(), p, (p / 2 if d.mean() > 0 else 1 - p / 2)


def main() -> None:
    data = {c: load(*c) for c in CELLS}
    if any(v is None for v in data.values()):
        print("incomplete:", [c for c, v in data.items() if v is None])
        return
    fmt = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
    md = ["# DIME training-budget check", ""]
    tex = ["\\begin{tabular}{lcccccc}", "\\toprule",
           "cell & DIME $1\\times$ & DIME $2\\times$ & DIME $4\\times$ & ours & ours $-$ DIME $4\\times$ ($p$) & "
           "DIME $4\\times - 1\\times$ ($p$) \\\\", "\\midrule"]
    p_one, n_pos = [], 0
    md.append("| cell | DIME 1x | 2x | 4x | ours | ours-1x (p) | ours-2x (p) | ours-4x (p) | 4x-1x (p) |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for (ds, n), R in data.items():
        a = {k: np.array([r[k] for r in R]) for k in R[0]}
        d1, p1, _ = paired(a["ours_supcond"], a["dime"])
        d2, p2, _ = paired(a["ours_supcond"], a["dime_x2"])
        d4, p4, p4one = paired(a["ours_supcond"], a["dime_x4"])
        g4, pg, _ = paired(a["dime_x4"], a["dime"])
        p_one.append(p4one)
        n_pos += d4 > 0
        name = f"{'METABRIC' if ds == 'metabric' else 'TCGA'} $\\Nl={n}$"
        md.append(f"| {ds} {n} | {a['dime'].mean():.3f} | {a['dime_x2'].mean():.3f} | {a['dime_x4'].mean():.3f} | "
                  f"{a['ours_supcond'].mean():.3f} | {d1:+.3f} ({p1:.3f}) | {d2:+.3f} ({p2:.3f}) | {d4:+.3f} ({p4:.3f}) | "
                  f"{g4:+.3f} ({pg:.3f}) |")
        tex.append(f"{name} & {a['dime'].mean():.3f} & {a['dime_x2'].mean():.3f} & {a['dime_x4'].mean():.3f} & "
                   f"{a['ours_supcond'].mean():.3f} & ${d4:+.3f}$ ({fmt(p4)}) & ${g4:+.3f}$ ({fmt(pg)}) \\\\")
    chi2 = -2 * np.sum(np.log(np.clip(p_one, 1e-300, 1)))
    fisher = float(stats.chi2.sf(chi2, df=8))
    ok = n_pos >= 3 and fisher < 0.05
    md[1:1] = [f"ours - DIME 4x positive in {n_pos}/4 cells; Fisher chi2={chi2:.2f}, df=8, p={fisher:.3g} -> "
               f"{'ADVANTAGE HOLDS WITH MORE DIME COMPUTE' if ok else 'advantage depends on DIME budget'}", ""]
    tex += ["\\bottomrule", "\\end{tabular}"]
    Path("experiments/aggregate/dime_budget_check.md").write_text("\n".join(md) + "\n")
    Path("paper/tables/dime_budget.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
