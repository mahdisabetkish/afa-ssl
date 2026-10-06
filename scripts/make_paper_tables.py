#!/usr/bin/env python
"""Generate every LaTeX table in the paper that is not produced by
scripts/aggregate.py, scripts/stats_extra.py or scripts/signal_alignment.py.

Earlier these were built by one-off snippets; this script is the single,
saved source for them, so `make eval` regenerates every number in the paper.

    python scripts/make_paper_tables.py [--no-dagger]

Tables written to paper/tables/:
  main_metabric_full.tex, main_tcga_full.tex   (Tables 2-3)
  eddi_dime_full.tex                           (Table 4)
  component_2x2.tex                            (Table 5)
  recovery_full.tex                            (Table 6)
  sensitivity.tex                              (Appendix, sensitivity)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

EXP = Path("experiments")
TAB = Path("paper/tables")
SUP = ["eig_sup", "er_sup", "ecd_sup", "eig_sup_ens"]


def load(prefix: str, ds: str, n: int, seeds=range(20)) -> dict[str, list[float]]:
    d: dict[str, list[float]] = {}
    for seed in seeds:
        p = EXP / f"{prefix}_{ds}_s{seed}_n{n}" / "summary.json"
        if not p.exists():
            continue
        for m, v in json.load(open(p))["methods"].items():
            d.setdefault(m, []).append(v["aucc"])
    return d


def fmt(mu: float, sd: float) -> str:
    return f"{mu:.3f} $\\pm$ {sd:.3f}"


# ---------------------------------------------------------------- Tables 2-3
MAIN_METHODS = ["rand", "static_sup", "eig_sup", "er_sup", "ecd_sup", "eig_sup_ens", "ens_rand",
                "ours_sslonly", "ours", "ours_supcond"]
MAIN_LABELS = {"rand": "Random", "static_sup": "Static $F$-score", "eig_sup": "EIG", "er_sup": "ER",
               "ecd_sup": "ECD", "eig_sup_ens": "EIG (sup. ensemble)", "ens_rand": "ensemble + random",
               "ours_sslonly": "SS-EIG, pretrained member only", "ours": "SS-EIG (pre-registered)",
               "ours_supcond": "\\textbf{ours}"}


def bh_significant_cells(method: str = "ours_supcond") -> set[tuple[str, int]]:
    """Cells where `method` beats the best supervised criterion at BH q<0.05 (scripts/stats_extra.py)."""
    p = EXP / "aggregate" / "stats_extra.json"
    if not p.exists():
        return set()
    return {(c["dataset"], 100000 if c["n_labels"] in (1574, 544) else c["n_labels"])
            for c in json.load(open(p))[method]["cells"] if c["significant_bh"] and c["mean_diff"] > 0}


def glioma_significant_cells() -> set[tuple[str, int]]:
    """Glioma cells significant after BH across its own four cells (scripts/gbmlgg_check.py)."""
    p = EXP / "aggregate" / "gbmlgg_check.json"
    if not p.exists():
        return set()
    return {("tcga_gbmlgg", n) for n in json.load(open(p))["significant_cells"]}


def main_table(ds: str, levels: list[tuple[str, int]], dagger: bool, with_sd: bool = True) -> str:
    data = {lv: load("main", ds, n) for lv, n in levels}
    best = {lv: max((m for m in MAIN_METHODS if m in d), key=lambda m: np.mean(d[m]))
            for lv, d in data.items()}
    sig = (glioma_significant_cells() if ds == "tcga_gbmlgg" else bh_significant_cells()) if dagger else set()
    if with_sd:
        head = ["Method & " + " & ".join(f"$N_l={lv}$" for lv, _ in levels) + " \\\\"]
    else:
        k = len(levels)
        head = [f" & \\multicolumn{{{k}}}{{c}}{{labeled rows $N_l$}} \\\\", f"\\cmidrule(lr){{2-{k + 1}}}",
                "Method & " + " & ".join(lv for lv, _ in levels) + " \\\\"]
    lines = ["\\begin{tabular}{l" + "c" * len(levels) + "}", "\\toprule", *head, "\\midrule"]
    for m in MAIN_METHODS:
        cells = []
        for lv, n in levels:
            d = data[lv]
            if m not in d:
                cells.append("--")
                continue
            cell = fmt(np.mean(d[m]), np.std(d[m])) if with_sd else f"{np.mean(d[m]):.3f}"
            if best[lv] == m:
                cell = "\\textbf{" + cell + "}"
            if dagger and m == "ours_supcond" and (ds, n) in sig:
                cell += "$^{\\dagger}$"
            cells.append(cell)
        lines.append(f"{MAIN_LABELS[m]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


# ---------------------------------------------------------------- Table 4
ED_METHODS = ["eig_sup_ens", "eddi", "eddi_ssl", "dime", "ours", "ours_supcond"]
ED_LABELS = {"eig_sup_ens": "best supervised (ensemble)", "eddi": "EDDI", "eddi_ssl": "EDDI (pool)",
             "dime": "DIME", "ours": "SS-EIG (pre-registered)", "ours_supcond": "\\textbf{ours}"}


def eddi_block(ds: str, levels=(25, 50, 100, 200)) -> str:
    data = {n: load("main", ds, n, seeds=range(5)) for n in levels}
    best = {n: max(ED_METHODS, key=lambda m: np.mean(data[n][m]) if m in data[n] else -1) for n in levels}
    lines = ["\\begin{tabular}{l" + "c" * len(levels) + "}", "\\toprule",
             "Method & " + " & ".join(f"$N_l={n}$" for n in levels) + " \\\\", "\\midrule"]
    for m in ED_METHODS:
        cells = []
        for n in levels:
            if m not in data[n]:
                cells.append("--")
                continue
            cell = fmt(np.mean(data[n][m]), np.std(data[n][m]))
            cells.append("\\textbf{" + cell + "}" if best[n] == m else cell)
        lines.append(f"{ED_LABELS[m]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def eddi_table() -> str:
    return ("\\textbf{METABRIC}\\\\[2pt]\n" + eddi_block("metabric") +
            "\\\\[8pt]\\textbf{TCGA breast RNA-seq}\\\\[2pt]\n" + eddi_block("tcga_brca"))


# ---------------------------------------------------------------- Table 5
def component_table() -> str:
    def mean(prefix, ds, m):
        return float(np.mean(load(prefix, ds, 50, seeds=range(5))[m]))
    lines = ["\\begin{tabular}{lcc}", "\\toprule",
             " & SSL (pretrained) conditional & Gaussian (labeled-only) conditional \\\\", "\\midrule"]
    for ds, name in (("metabric", "METABRIC"), ("tcga_brca", "TCGA breast RNA-seq")):
        ens_ssl, ens_g = mean("ablation", ds, "ours"), mean("ablation", ds, "ours_supcond")
        sup_ssl, sup_g = mean("ablation", ds, "sup_eig_sslcond"), mean("main", ds, "eig_sup")
        cells = {"es": ens_ssl, "eg": ens_g, "ss": sup_ssl, "sg": sup_g}
        top = max(cells, key=cells.get)
        b = {k: (f"\\textbf{{{v:.3f}}}" if k == top else f"{v:.3f}") for k, v in cells.items()}
        lines.append(f"\\multicolumn{{3}}{{l}}{{\\textit{{{name}, $N_l=50$}}}} \\\\")
        lines.append(f"Ensemble predictive & {b['es']} (SS-EIG, pre-registered) & {b['eg']} (ours) \\\\")
        lines.append(f"Supervised predictive & {b['ss']} & {b['sg']} (standard EIG baseline) \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


# ---------------------------------------------------------------- Table 6
REC_METHODS = ["ours_supcond", "ours", "ours_sslonly", "eig_sup_ens", "eig_sup", "static_sup", "unc", "rand"]
REC_LABELS = {"ours_supcond": "ours", "ours": "SS-EIG", "ours_sslonly": "SS-EIG, pretrained only", "eig_sup_ens": "EIG (sup. ensemble)",
              "eig_sup": "EIG (sup.)", "static_sup": "Static $F$-score", "unc": "Uncertainty (sup.)",
              "rand": "Random"}


def recovery_table() -> str:
    lines = []
    for ds, dsname in (("synth_scm", "synthetic SCM"), ("metabric_semisynth", "METABRIC semi-synthetic")):
        for n in (50, 200):
            d: dict[str, dict[str, list[float]]] = {}
            for seed in range(5):
                s = json.load(open(EXP / f"main_{ds}_s{seed}_n{n}" / "summary.json"))
                for m, v in s["methods"].items():
                    if "precision_at_k" in v:
                        d.setdefault(m, {}).setdefault("prec", []).append(v["precision_at_k"])
                        d.setdefault(m, {}).setdefault("red", []).append(v["redundancy_at_k"])
            best = max((m for m in REC_METHODS if m in d), key=lambda m: np.mean(d[m]["prec"]))
            for m in REC_METHODS:
                if m not in d:
                    continue
                pr, rd = np.array(d[m]["prec"]), np.array(d[m]["red"])
                p = f"{pr.mean():.3f} $\\pm$ {pr.std():.3f}"
                p = "\\textbf{" + p + "}" if m == best else p
                lines.append(f"{dsname} & {n} & {REC_LABELS[m]} & {p} & "
                             f"{rd.mean():.3f} $\\pm$ {rd.std():.3f} \\\\")
            lines.append("\\addlinespace")
    return ("\\begin{tabular}{llcccc}\n\\toprule\ndataset & $N_l$ & method & precision@$k$ & "
            "redundancy@$k$ \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")


# ---------------------------------------------------------------- sensitivity (appendix)
SENS = [("capacity", "small", "encoder hidden 64 / latent 32"),
        ("capacity", "large", "encoder hidden 256 / latent 128"),
        ("poolfrac", "f0.1", "10\\% of pool for pretraining"),
        ("poolfrac", "f0.3", "30\\% of pool for pretraining"),
        ("objective", "mse", "fixed-variance (MSE) objective"),
        ("objective", "unifmask", "uniform mask distribution")]


def sensitivity_table() -> str:
    base = np.array(load("main", "metabric", 50, seeds=range(5))["ours"])
    lines = ["\\begin{tabular}{lccc}", "\\toprule", "variant & AUCC & difference & $p$ \\\\", "\\midrule",
             f"default & {base.mean():.3f} $\\pm$ {base.std():.3f} & --- & --- \\\\"]
    for exp, tag, label in SENS:
        vals = []
        for seed in range(5):
            p = EXP / f"{exp}_metabric_s{seed}_n50_{tag}" / "summary.json"
            if p.exists():
                vals.append(json.load(open(p))["methods"]["ours"]["aucc"])
        v = np.array(vals)
        pv = stats.ttest_rel(v, base).pvalue
        lines.append(f"{label} & {v.mean():.3f} $\\pm$ {v.std():.3f} & {v.mean() - base.mean():+.3f} & {pv:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dagger", action="store_true",
                    help="omit BH-significance daggers (reproduces the tables as first published)")
    args = ap.parse_args()
    dagger = not args.no_dagger
    TAB.mkdir(parents=True, exist_ok=True)
    outputs = {
        "main_metabric_full.tex": main_table("metabric", [("25", 25), ("50", 50), ("100", 100), ("200", 200),
                                                           ("400", 400), ("all", 100000)], dagger),
        "main_tcga_full.tex": main_table("tcga_brca", [("25", 25), ("50", 50), ("100", 100), ("200", 200),
                                                        ("all", 100000)], dagger),
        "main_gbmlgg_full.tex": main_table("tcga_gbmlgg", [("25", 25), ("50", 50), ("100", 100), ("200", 200)],
                                           dagger),
        "main_metabric_mean.tex": main_table("metabric", [("25", 25), ("50", 50), ("100", 100), ("200", 200),
                                                           ("400", 400), ("all", 100000)], dagger, with_sd=False),
        "main_tcga_mean.tex": main_table("tcga_brca", [("25", 25), ("50", 50), ("100", 100), ("200", 200),
                                                        ("all", 100000)], dagger, with_sd=False),
        "main_gbmlgg_mean.tex": main_table("tcga_gbmlgg", [("25", 25), ("50", 50), ("100", 100), ("200", 200)],
                                           dagger, with_sd=False),
        "eddi_dime_full.tex": eddi_table(),
        "component_2x2.tex": component_table(),
        "recovery_full.tex": recovery_table(),
        "sensitivity.tex": sensitivity_table(),
    }
    for name, text in outputs.items():
        (TAB / name).write_text(text)
        print("wrote", TAB / name)


if __name__ == "__main__":
    main()
