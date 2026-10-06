#!/usr/bin/env python
"""Accuracy at small feature budgets and the number of features needed to match it.

AUCC averages accuracy over budgets 1..B, which understates what the gap means in measurements. For
every primary cell and every glioma cell this reports, from the stored acquisition curves, the mean
test accuracy of the proposed method and of the best supervised criterion (chosen per cell by mean
AUCC, as in the main table) after 3, 5 and 10 purchases, the number of seeds on which the proposed
method is ahead at 5, and the smallest budget at which the best supervised criterion's mean curve
reaches the proposed method's mean accuracy at 5. Descriptive only: no new test is applied.
Writes experiments/aggregate/practical_effect.json and paper/tables/practical_effect.tex.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EXP = Path("experiments")
SUP = ["eig_sup", "er_sup", "ecd_sup", "eig_sup_ens"]
OURS = "ours_supcond"
BUDGETS = (3, 5, 10)
K_MATCH = 5
COHORTS = [("metabric", "METABRIC", (25, 50, 100, 200, 400, 100000)),
           ("tcga_brca", "TCGA breast RNA-seq", (25, 50, 100, 200, 100000)),
           ("tcga_gbmlgg", "TCGA glioma (pre-specified external cohort)", (25, 50, 100, 200)),
           ("uci_har", "Human activity, smartphone (pre-specified non-omics dataset)", (25, 50, 100, 200))]


def load(ds: str, n: int) -> list[dict]:
    runs = []
    for s in range(20):
        f = EXP / f"main_{ds}_s{s}_n{n}" / "summary.json"
        if f.exists():
            runs.append(json.load(open(f))["methods"])
    return runs


def signed(x: float) -> str:
    """Signed difference to three decimals; a value that rounds to zero is shown without a sign."""
    return "$0.000$" if abs(x) < 5e-4 else f"${x:+.3f}$"


def features_to_match(curve: np.ndarray, target: float) -> int | None:
    """Smallest budget k >= 1 at which the curve reaches the target, or None within the budget."""
    hit = np.where(curve[1:] >= target - 1e-9)[0]
    return int(hit[0]) + 1 if len(hit) else None


def cell(ds: str, n: int) -> dict:
    runs = load(ds, n)
    best = max(SUP, key=lambda m: np.mean([r[m]["aucc"] for r in runs]))
    o = np.array([r[OURS]["acc_curve"] for r in runs])
    b = np.array([r[best]["acc_curve"] for r in runs])
    om, bm = o.mean(0), b.mean(0)
    return {"dataset": ds, "n_labels": n, "seeds": len(runs), "best_sup": best,
            "acc_ours": {k: float(om[k]) for k in BUDGETS}, "acc_sup": {k: float(bm[k]) for k in BUDGETS},
            "diff": {k: float(om[k] - bm[k]) for k in BUDGETS},
            "seeds_ahead_at_5": int((o[:, K_MATCH] > b[:, K_MATCH]).sum()),
            "sup_features_to_match_5": features_to_match(bm, om[K_MATCH]),
            "budget": int(o.shape[1] - 1)}


def main() -> None:
    out, tex = [], ["\\begin{tabular}{lccccccc}", "\\toprule",
                    " & \\multicolumn{4}{c}{accuracy after 5 features} & features & \\multicolumn{2}{c}{difference after} \\\\",
                    "\\cmidrule(lr){2-5}\\cmidrule(lr){7-8}",
                    "$\\Nl$ & ours & best sup. & difference & seeds ahead & to match & 3 & 10 \\\\", "\\midrule"]
    for ds, name, ns in COHORTS:
        tex.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{name}}}}} \\\\")
        for n in ns:
            c = cell(ds, n)
            out.append(c)
            m = c["sup_features_to_match_5"]
            match = str(m) if m is not None else f"$>{c['budget']}$"
            tex.append(f"{'all' if n == 100000 else n} & {c['acc_ours'][5]:.3f} & {c['acc_sup'][5]:.3f} & "
                       f"{signed(c['diff'][5])} & {c['seeds_ahead_at_5']}/{c['seeds']} & {match} & "
                       f"{signed(c['diff'][3])} & {signed(c['diff'][10])} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    Path("experiments/aggregate/practical_effect.json").write_text(json.dumps(out, indent=1))
    Path("paper/tables/practical_effect.tex").write_text("\n".join(tex) + "\n")
    for c in out:
        print(f"{c['dataset']:12s} {c['n_labels']:6d} s={c['seeds']:2d} {c['best_sup']:11s} "
              f"acc5 {c['acc_ours'][5]:.3f}/{c['acc_sup'][5]:.3f} ({c['diff'][5]:+.3f}, {c['seeds_ahead_at_5']}/{c['seeds']}) "
              f"match {c['sup_features_to_match_5']}  d3 {c['diff'][3]:+.3f} d10 {c['diff'][10]:+.3f}")


if __name__ == "__main__":
    main()
