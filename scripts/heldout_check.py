#!/usr/bin/env python
"""Apply the promotion rule (fixed before inspecting results) to the
held-out cells: is the Gaussian-conditional variant (`ours_supcond`) clearly
better than the registered SS-EIG (`ours`) on data not used to identify it?

Rule: promote iff (a) variant mean >= registered mean in >= 8 of 12 cells,
(b) Fisher's combined one-sided test (variant better) p < 0.05, and
(c) no cell significantly worse after Benjamini-Hochberg across the 12.

Also reports the variant against the best supervised criterion in the same
cells. Writes experiments/aggregate/heldout_check.md and
paper/tables/heldout_check.tex.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

CELLS = [(ds, n) for ds in ("metabric_er", "metabric_her2", "metabric_intclust", "metabric_grade",
                            "synth_scm", "metabric_semisynth") for n in (50, 200)]
NAMES = {"metabric_er": "ER status", "metabric_her2": "HER2 status", "metabric_intclust": "integrative cluster",
         "metabric_grade": "histological grade", "synth_scm": "synthetic SCM",
         "metabric_semisynth": "semi-synthetic"}
SUP = ["eig_sup", "er_sup", "ecd_sup", "eig_sup_ens"]


def cell(ds: str, n: int):
    v, o, sup = [], [], {m: [] for m in SUP}
    for seed in range(5):
        s = json.load(open(Path("experiments") / f"main_{ds}_s{seed}_n{n}" / "summary.json"))["methods"]
        if "ours_supcond" not in s:
            return None
        v.append(s["ours_supcond"]["aucc"])
        o.append(s["ours"]["aucc"])
        for m in SUP:
            if m in s:
                sup[m].append(s[m]["aucc"])
    v, o = np.array(v), np.array(o)
    best = max((m for m in SUP if sup[m]), key=lambda m: np.mean(sup[m]))
    b = np.array(sup[best])
    d = v - o
    t, p2 = stats.ttest_rel(v, o)
    p_one = p2 / 2 if d.mean() > 0 else 1 - p2 / 2          # one-sided, H1: variant better
    p_worse = p2 / 2 if d.mean() < 0 else 1 - p2 / 2        # one-sided, H1: variant worse
    return {"ds": ds, "n": n, "variant": v.mean(), "registered": o.mean(), "diff": d.mean(),
            "p_two": p2, "p_better": p_one, "p_worse": p_worse,
            "best_sup": best, "sup": b.mean(), "diff_vs_sup": (v - b).mean(),
            "p_vs_sup": stats.ttest_rel(v, b).pvalue, "wins": int((d > 0).sum()),
            "reg_vs_sup": (o - b).mean(), "p_reg_vs_sup": stats.ttest_rel(o, b).pvalue}


def main() -> None:
    rows = [cell(ds, n) for ds, n in CELLS]
    missing = [c for c, r in zip(CELLS, rows) if r is None]
    if missing:
        print(f"not all held-out cells are complete yet: {len(missing)} missing, e.g. {missing[:3]}")
        return
    n_ge = sum(r["diff"] >= 0 for r in rows)
    chi2 = -2 * np.sum(np.log(np.clip([r["p_better"] for r in rows], 1e-300, 1)))
    fisher_p = stats.chi2.sf(chi2, df=2 * len(rows))
    rej_worse, q_worse, _, _ = multipletests([r["p_worse"] for r in rows], alpha=0.05, method="fdr_bh")
    a, b, c = n_ge >= 8, fisher_p < 0.05, not rej_worse.any()
    verdict = "PROMOTE" if (a and b and c) else "KEEP SS-EIG AS REGISTERED"

    out = ["# Held-out check", "",
           f"(a) variant >= registered in {n_ge}/12 cells (need >= 8): {'pass' if a else 'fail'}",
           f"(b) Fisher combined one-sided p = {fisher_p:.3g} (chi2={chi2:.2f}, df={2*len(rows)}) (need < 0.05): "
           f"{'pass' if b else 'fail'}",
           f"(c) cells significantly worse after BH: {int(rej_worse.sum())} (need 0): {'pass' if c else 'fail'}",
           f"VERDICT: {verdict}", "",
           "| label | N_l | variant | registered | diff | p (two-sided) | wins | vs best sup. | p |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {NAMES[r['ds']]} | {r['n']} | {r['variant']:.3f} | {r['registered']:.3f} | "
                   f"{r['diff']:+.3f} | {r['p_two']:.3f} | {r['wins']}/5 | {r['diff_vs_sup']:+.3f} | "
                   f"{r['p_vs_sup']:.3f} |")
    text = "\n".join(out)
    Path("experiments/aggregate/heldout_check.md").write_text(text + "\n")
    print(text)

    lines = ["\\begin{tabular}{lrcccccc}", "\\toprule",
             " & & & & \\multicolumn{2}{c}{ours $-$ SS-EIG} & ours $-$ best sup. & SS-EIG $-$ best sup. \\\\",
             "\\cmidrule(lr){5-6}",
             "label & $N_l$ & ours & SS-EIG & diff. & $p$ & diff.\\ ($p$) & diff.\\ ($p$) \\\\",
             "\\midrule"]
    for r in rows:
        lines.append(f"{NAMES[r['ds']]} & {r['n']} & {r['variant']:.3f} & {r['registered']:.3f} & "
                     f"${r['diff']:+.3f}$ & {r['p_two']:.3f} & ${r['diff_vs_sup']:+.3f}$ ({r['p_vs_sup']:.3f}) & "
                     f"${r['reg_vs_sup']:+.3f}$ ({r['p_reg_vs_sup']:.3f}) \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    Path("paper/tables/heldout_check.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
