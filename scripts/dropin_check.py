#!/usr/bin/env python
"""Apply the decision rule (fixed before any run): does the proposed predictor, dropped into other
acquisition methods with everything else unchanged, improve them?

Pairs (variant vs original; only the predictor differs): ER, ECD, DIME. Cells: METABRIC and TCGA at
N_l in {25, 50}, seeds 0-4. Per pair, "improves" iff the mean AUCC difference is > 0 in >= 3 of 4
cells AND Fisher's combined one-sided test (variant better) over the 4 cells gives p < 0.05.
Per-cell p-values are also reported with Benjamini-Hochberg across the 12 comparisons. The EIG pair
(the proposed method vs supervised EIG) is shown alongside for reference and is not part of the
family. Writes experiments/aggregate/dropin_check.md and paper/tables/dropin.tex.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

CELLS = [("metabric", 25), ("metabric", 50), ("tcga_brca", 25), ("tcga_brca", 50)]
PAIRS = [("ens_er_g", "er_sup", "ER"), ("ens_ecd_g", "ecd_sup", "ECD"), ("dime_ens", "dime", "DIME")]
REFERENCE = ("ours_supcond", "eig_sup", "EIG")
SEEDS = range(5)


def cell(ds: str, n: int, a: str, b: str):
    x, y = [], []
    for s in SEEDS:
        f = Path("experiments") / f"main_{ds}_s{s}_n{n}" / "summary.json"
        m = json.load(open(f))["methods"] if f.exists() else {}
        if a not in m or b not in m:
            return None
        x.append(m[a]["aucc"])
        y.append(m[b]["aucc"])
    x, y = np.array(x), np.array(y)
    d = x - y
    p2 = stats.ttest_rel(x, y).pvalue
    p_better = p2 / 2 if d.mean() > 0 else 1 - p2 / 2
    return {"variant": x.mean(), "original": y.mean(), "diff": d.mean(), "p": p2, "p_better": p_better,
            "wins": int((d > 0).sum())}


def main() -> None:
    res = {(a, ds, n): cell(ds, n, a, b) for a, b, _ in PAIRS + [REFERENCE] for ds, n in CELLS}
    missing = [k for k, v in res.items() if v is None and k[0] != REFERENCE[0]]
    if missing:
        print(f"incomplete: {len(missing)} (pair, cell) combinations missing, e.g. {missing[:3]}")
        return
    fam = [(a, ds, n) for a, _, _ in PAIRS for ds, n in CELLS]
    rej, q, _, _ = multipletests([res[k]["p"] for k in fam], alpha=0.05, method="fdr_bh")
    qv = dict(zip(fam, q))
    md = ["# Drop-in check", ""]
    verdicts = {}
    for a, b, name in PAIRS + [REFERENCE]:
        rows = [res[(a, ds, n)] for ds, n in CELLS]
        n_pos = sum(r["diff"] > 0 for r in rows)
        chi2 = -2 * np.sum(np.log(np.clip([r["p_better"] for r in rows], 1e-300, 1)))
        fisher = float(stats.chi2.sf(chi2, df=2 * len(rows)))
        ok = n_pos >= 3 and fisher < 0.05
        verdicts[name] = (n_pos, chi2, fisher, ok)
        tag = "" if (a, b, name) != REFERENCE else " (reference, not in the family)"
        md.append(f"{name}{tag}: positive in {n_pos}/4 cells; Fisher chi2={chi2:.2f}, df=8, p={fisher:.3g} -> "
                  f"{'IMPROVES' if ok else 'does not meet the rule'}")
    md += ["", "| method | cell | with ours predictor | original | diff | p | q (BH, 12) |", "|---|---|---|---|---|---|---|"]
    for a, b, name in PAIRS + [REFERENCE]:
        for ds, n in CELLS:
            r = res[(a, ds, n)]
            qs = f"{qv[(a, ds, n)]:.3f}" if (a, ds, n) in qv else "--"
            md.append(f"| {name} | {ds} {n} | {r['variant']:.3f} | {r['original']:.3f} | {r['diff']:+.3f} | "
                      f"{r['p']:.3f} | {qs} |")
    Path("experiments/aggregate/dropin_check.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    head = " & ".join(f"{'METABRIC' if ds == 'metabric' else 'TCGA'} ${n}$" for ds, n in CELLS)
    tex = ["\\begin{tabular}{l" + "c" * len(CELLS) + "c}", "\\toprule",
           "method & " + head.replace("$25$", "$\\Nl{=}25$").replace("$50$", "$\\Nl{=}50$") +
           " & combined $p$ \\\\", "\\midrule"]
    for a, b, name in PAIRS + [REFERENCE]:
        cells = []
        for ds, n in CELLS:
            r = res[(a, ds, n)]
            pt = "$<$0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
            cells.append(f"${r['diff']:+.3f}$ ({pt})")
        n_pos, chi2, fisher, ok = verdicts[name]
        label = name if (a, b, name) != REFERENCE else "EIG (the proposed method)"
        if fisher < 1e-3:
            mant, ex = f"{fisher:.1e}".split("e")
            fp = f"${mant}\\times10^{{{int(ex)}}}$"
        else:
            fp = f"{fisher:.3f}"
        tex.append(f"{label} & " + " & ".join(cells) + f" & {fp} \\\\")
        if name == "DIME":
            tex.append("\\midrule")
    tex += ["\\bottomrule", "\\end{tabular}"]
    Path("paper/tables/dropin.tex").write_text("\n".join(tex) + "\n")


if __name__ == "__main__":
    main()
