"""Collect experiments/*/summary.json into tidy tables and run the paired tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from afa_ssl.eval.stats import paired_summary

METHOD_LABELS = {
    "rand": "Random", "static_sup": "Static F-score (sup.)", "ssl_static": "SSL + static F-score", "unc": "Uncertainty (sup.)", "eig_sup": "EIG (sup.)", "er_sup": "ER (sup.)",
    "ecd_sup": "ECD (sup.)", "eig_sup_mfa": "EIG (sup., MFA cond.)", "ssl_rand": "SSL + random",
    "ssl_unc": "SSL + uncertainty", "ours": "SS-EIG (pre-registered)", "ours_sslonly": "SS-EIG, pretrained member only", "ours_supcond": "ours", "ens_er_g": "ER, ours predictor", "ens_ecd_g": "ECD, ours predictor", "dime_ens": "DIME, ours predictor", "dime_x2": "DIME, 2x training", "dime_x4": "DIME, 4x training",
    "ours_w05": "SS-EIG, fixed 1/2 mixture",
    "eddi": "EDDI", "eddi_eig": "EDDI (EIG)", "eddi_rand": "EDDI model + random",
    "eddi_ssl": "EDDI, pool-trained VAE", "eddi_ssl_eig": "EDDI pool (EIG)",
    "eddi_ssl_rand": "EDDI pool + random", "dime": "DIME",
    "ours_div": "SS-EIG + diversity factor", "ours_supcond_div": "ours + diversity factor", "ens_rand": "ensemble + random", "ens_static": "ensemble + static F-score", "ens_unc": "ensemble + uncertainty",
    "ens_er": "ensemble + ER", "ens_ecd": "ensemble + ECD", "eig_sup_ens": "EIG (sup. ensemble)",
    "sup_ens_rand": "sup. ensemble + random", "ours_frozen": "SS-EIG, frozen encoder", "ssl_frozen_rand": "SSL frozen + random", "ssl_er": "SSL + ER", "ssl_ecd": "SSL + ECD",
    "ssl_eig_supcond": "pretrained member only, Gaussian cond.", "sup_eig_sslcond": "sup. head, SSL cond.",
    "ssl_eig_gausspool": "pretrained member only, Gaussian pool cond.", "ours_costblind": "SS-EIG cost-blind",
    "eig_sup_costblind": "EIG (sup.) cost-blind",
}
SUP_CRITERIA = ["eig_sup", "er_sup", "ecd_sup", "eig_sup_mfa", "eig_sup_ens"]


def load_runs(root: str | Path = "experiments", experiment: str | None = None) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(root).glob("*/summary.json")):
        s = json.load(open(p))
        exp = s["run_id"].split("_")[0]
        if experiment and exp != experiment:
            continue
        tag = s["run_id"].split(f"_n{s['n_labels']}")[-1].lstrip("_") if f"_n{s['n_labels']}" in s["run_id"] else ""
        for m, v in s["methods"].items():
            r = {"experiment": exp, "tag": tag, "run_id": s["run_id"], "dataset": s["dataset"],
                 "seed": s["seed"], "n_labels": s["n_labels"], "method": m, "aucc": v["aucc"],
                 "aucc_steps": v.get("aucc_uniform_steps", v["aucc"]), "acc_final": v["acc_final"],
                 "acc_full": v["acc_full"], "seconds": v["seconds"], "acc_curve": v["acc_curve"],
                 "cost_budget": s["cost_budget"], "runtime_s": s["runtime_s"]}
            for k in ("precision_at_k", "group_coverage_at_k", "redundancy_at_k", "redundancy_all", "mixture_weight"):
                if k in v:
                    r[k] = v[k]
            rows.append(r)
    return pd.DataFrame(rows)


def mean_std_table(df: pd.DataFrame, metric: str = "aucc") -> pd.DataFrame:
    g = df.groupby(["dataset", "n_labels", "method"])[metric]
    out = g.agg(["mean", "std", "count"]).reset_index()
    return out


def paired_vs(df: pd.DataFrame, ref: str, others: list[str], metric: str = "aucc") -> pd.DataFrame:
    """Paired tests of `ref` against each method in `others` within (dataset, n_labels)."""
    rows = []
    for (ds, n), sub in df.groupby(["dataset", "n_labels"]):
        piv = sub.pivot_table(index="seed", columns="method", values=metric)
        if ref not in piv:
            continue
        for o in others:
            if o not in piv:
                continue
            both = piv[[ref, o]].dropna()
            if len(both) < 2:
                continue
            st = paired_summary(both[ref].to_numpy(), both[o].to_numpy())
            rows.append({"dataset": ds, "n_labels": n, "ref": ref, "other": o, **st})
    return pd.DataFrame(rows)


def best_supervised(df: pd.DataFrame, metric: str = "aucc") -> dict[tuple[str, int], str]:
    """Per (dataset, n_labels), the supervised criterion with the highest mean (favours the baseline)."""
    out = {}
    for (ds, n), sub in df[df.method.isin(SUP_CRITERIA)].groupby(["dataset", "n_labels"]):
        out[(ds, n)] = sub.groupby("method")[metric].mean().idxmax()
    return out


def fmt_ms(m: float, s: float, digits: int = 3) -> str:
    return f"{m:.{digits}f} $\\pm$ {s:.{digits}f}" if np.isfinite(s) else f"{m:.{digits}f}"


def latex_table(df: pd.DataFrame, dataset: str, methods: list[str], metric: str = "aucc",
                bold_best: bool = True) -> str:
    sub = df[(df.dataset == dataset) & (df.method.isin(methods))]
    levels = sorted(sub.n_labels.unique())
    ms = sub.groupby(["n_labels", "method"])[metric].agg(["mean", "std"])
    lines = ["\\begin{tabular}{l" + "c" * len(levels) + "}", "\\toprule",
             "Method & " + " & ".join(f"$N_l={n}$" for n in levels) + " \\\\", "\\midrule"]
    best = {n: ms.xs(n, level="n_labels")["mean"].idxmax() for n in levels}
    for m in methods:
        cells = []
        for n in levels:
            if (n, m) in ms.index:
                mu, sd = ms.loc[(n, m)]
                cell = fmt_ms(mu, sd)
                if bold_best and best[n] == m:
                    cell = "\\textbf{" + cell + "}"
                cells.append(cell)
            else:
                cells.append("--")
        lines.append(f"{METHOD_LABELS.get(m, m)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)
