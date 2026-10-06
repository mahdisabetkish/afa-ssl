#!/usr/bin/env python
"""Statistical additions beyond the per-cell paired t-tests already in
scripts/aggregate.py (statistics bundle):

  1. Benjamini-Hochberg correction across the primary claim family (ours vs
     best supervised criterion, at every (dataset, n_labels) cell of the two
     primary cohorts).
  2. A post-hoc power analysis: the minimum AUCC difference each cell's seed
     count could detect at 80% power, alpha=0.05, paired t-test, so a reader
     can judge whether a non-significant cell reflects a genuinely small
     effect or an underpowered test.
  3. A pooled test across the two primary cohorts (Fisher's combined
     probability test on the primary-family p-values, and a fixed-effect
     inverse-variance-weighted pooled Cohen's d_z) to answer "does the
     advantage exist across cohorts" with one number rather than reading
     eleven separate cells.

No new experiments are run; this reads experiments/aggregate/all_runs.pkl
(built by scripts/aggregate.py) and writes paper/tables/stats_extra_*.tex.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestPower

OUT = "experiments/aggregate"
TAB = "paper/tables"

SUP_CRITERIA = ["eig_sup", "er_sup", "ecd_sup", "eig_sup_ens"]
PRIMARY_CELLS = [("metabric", n) for n in (25, 50, 100, 200, 400, 1574)] + \
               [("tcga_brca", n) for n in (25, 50, 100, 200, 544)]


def paired_cell(df: pd.DataFrame, ds: str, n: int, method: str = "ours"):
    sub = df[(df.experiment == "main") & (df.dataset == ds) & (df.n_labels == n)]
    piv = sub.pivot_table(index="seed", columns="method", values="aucc")
    sups = {m: piv[m] for m in SUP_CRITERIA if m in piv}
    if not sups or method not in piv:
        return None
    best = max(sups, key=lambda m: sups[m].mean())
    both = pd.concat([piv[method], sups[best]], axis=1, keys=["a", "b"]).dropna()
    if len(both) < 2:
        return None
    d = both["a"] - both["b"]
    t, p = stats.ttest_rel(both["a"], both["b"])
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else 0.0
    return {"dataset": ds, "n_labels": n, "best_sup": best, "n_seeds": len(both),
            "mean_diff": d.mean(), "p": p, "dz": dz}


METHODS = {"ours_supcond": "proposed (headline)", "ours": "SS-EIG (pre-registered)"}


def family(df: pd.DataFrame, method: str) -> dict:
    rows = [paired_cell(df, ds, n, method) for ds, n in PRIMARY_CELLS]
    res = pd.DataFrame([r for r in rows if r is not None])
    reject, qvals, _, _ = multipletests(res["p"].to_numpy(), alpha=0.05, method="fdr_bh")
    res["q_bh"], res["significant_bh"] = qvals, reject
    res["n_display"] = res["n_labels"].apply(lambda n: "all" if n in (1574, 544) else n)
    power_calc = TTestPower()
    res["mde_dz"] = res["n_seeds"].map(lambda n: power_calc.solve_power(
        effect_size=None, nobs=n, alpha=0.05, power=0.8, alternative="two-sided"))
    res["adequately_powered"] = res["dz"].abs() >= res["mde_dz"]
    # Fisher's combined test on one-sided p-values in the direction of each observed effect
    one_sided = res.apply(lambda r: r["p"] / 2 if r["mean_diff"] > 0 else 1 - r["p"] / 2, axis=1)
    chi2 = float(-2 * np.sum(np.log(one_sided.clip(lower=1e-300))))
    fisher_p = float(stats.chi2.sf(chi2, df=2 * len(res)))
    # seed-count-weighted pooled d_z (simple fixed-effect approximation)
    w = res["n_seeds"].to_numpy(dtype=float)
    pooled = float(np.sum(w * res["dz"]) / np.sum(w))
    se = float(1.0 / np.sqrt(np.sum(w)))
    return {"res": res, "chi2": chi2, "df": 2 * len(res), "fisher_p": fisher_p, "pooled_dz": pooled,
            "ci": (pooled - 1.96 * se, pooled + 1.96 * se), "n_raw": int((res["p"] < 0.05).sum()),
            "n_bh": int(res["significant_bh"].sum()), "n_underpowered": int((~res["adequately_powered"]).sum())}


def main() -> None:
    df = pd.read_pickle(f"{OUT}/all_runs.pkl")
    fams = {m: family(df, m) for m in METHODS}

    head, reg = fams["ours_supcond"]["res"], fams["ours"]["res"].set_index(["dataset", "n_labels"])
    lines = ["\\begin{tabular}{llrrrrrrr}", "\\toprule",
             " & & \\multicolumn{4}{c}{proposed} & \\multicolumn{3}{c}{SS-EIG (pre-registered)} \\\\",
             "\\cmidrule(lr){3-6}\\cmidrule(lr){7-9}",
             "dataset & $N_l$ & seeds & diff & $p$ & $q$ (BH) & diff & $p$ & $q$ (BH) \\\\", "\\midrule"]
    for _, r in head.iterrows():
        g = reg.loc[(r["dataset"], r["n_labels"])]
        name = {"metabric": "METABRIC", "tcga_brca": "TCGA"}[r["dataset"]]
        star = lambda q: "$^{*}$" if q < 0.05 else ""
        lines.append(f"{name} & {r['n_display']} & {r['n_seeds']} & {r['mean_diff']:+.3f} & {r['p']:.4f} & "
                     f"{r['q_bh']:.4f}{star(r['q_bh'])} & {g['mean_diff']:+.3f} & {g['p']:.4f} & "
                     f"{g['q_bh']:.4f}{star(g['q_bh'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"{TAB}/bh_correction.tex", "w").write("\n".join(lines) + "\n")

    seed_counts = sorted(set(head["n_seeds"]) | set(reg["n_seeds"]))
    power_calc = TTestPower()
    mde = {n: power_calc.solve_power(effect_size=None, nobs=n, alpha=0.05, power=0.8, alternative="two-sided")
           for n in seed_counts}
    lines = ["\\begin{tabular}{lc}", "\\toprule", "seeds & min.\\ detectable $|d_z|$ (80\\% power) \\\\",
             "\\midrule"] + [f"{n} & {mde[n]:.2f} \\\\" for n in seed_counts] + ["\\bottomrule", "\\end{tabular}"]
    open(f"{TAB}/power_analysis.tex", "w").write("\n".join(lines) + "\n")

    md = ["# Statistics extras", "",
          "Primary claim family: (dataset, n_labels) cells on the two primary cohorts, method vs best "
          "supervised criterion (paired by seed).", ""]
    js = {}
    for m, label in METHODS.items():
        f = fams[m]
        r = f["res"]
        md += [f"## {m} ({label})", "",
               f"- Raw p<0.05: {f['n_raw']}/{len(r)}. After Benjamini-Hochberg (q<0.05): {f['n_bh']}/{len(r)}.",
               f"- Fisher combined (one-sided, direction of effect): chi2={f['chi2']:.2f}, df={f['df']}, "
               f"p={f['fisher_p']:.2e}.",
               f"- Seed-weighted pooled d_z: {f['pooled_dz']:.3f} (approx. 95% CI [{f['ci'][0]:.3f}, {f['ci'][1]:.3f}]).",
               f"- Cells below their own 80%-power threshold: {f['n_underpowered']}/{len(r)}.", "",
               r[["dataset", "n_labels", "best_sup", "n_seeds", "mean_diff", "dz", "p", "q_bh", "significant_bh",
                  "mde_dz", "adequately_powered"]].to_string(index=False), ""]
        js[m] = {k: v for k, v in f.items() if k != "res"}
        js[m]["cells"] = r.drop(columns=["n_display"]).to_dict(orient="records")
    md += ["Minimum detectable |d_z| at 80% power: " + ", ".join(f"{n} seeds -> {mde[n]:.2f}" for n in seed_counts)]
    open(f"{OUT}/stats_extra.md", "w").write("\n".join(md) + "\n")
    import json
    json.dump(js, open(f"{OUT}/stats_extra.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("\n".join(md))


if __name__ == "__main__":
    main()
