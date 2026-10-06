#!/usr/bin/env python
"""Generate every paper figure from experiments/. No manual editing anywhere."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from afa_ssl.eval.aggregate import METHOD_LABELS, load_runs  # noqa: E402
from afa_ssl.eval.metrics import accuracy_at_cost_grid  # noqa: E402

FIG = Path("paper/figures")
# fixed colour per method (validated palette); solid = uses the SSL encoder, dashed = supervised only
STYLE = {
    "ours": ("#2a78d6", "--"), "eig_sup": ("#eb6834", "--"), "ssl_rand": ("#1baf7a", "-"),
    "rand": ("#4a3aa7", "--"), "er_sup": ("#eda100", "--"), "ecd_sup": ("#e87ba4", "--"),
    "unc": ("#008300", "--"), "ssl_unc": ("#008300", "-"), "eig_sup_mfa": ("#eb6834", ":"),
    "ssl_er": ("#eda100", "-"), "ssl_ecd": ("#e87ba4", "-"), "ssl_eig_supcond": ("#eb6834", "-."),
    "sup_eig_sslcond": ("#4a3aa7", "-."), "ssl_eig_gausspool": ("#1baf7a", "-."),
    "ours_costblind": ("#2a78d6", ":"), "eig_sup_costblind": ("#eb6834", ":"),
    "static_sup": ("#4a3aa7", ":"), "ssl_static": ("#4a3aa7", "-"),
    "ours_frozen": ("#2a78d6", "--"), "ssl_frozen_rand": ("#1baf7a", "--"),
    "eig_sup_ens": ("#eb6834", "-"), "ens_rand": ("#1baf7a", "-"), "ours_sslonly": ("#2a78d6", "-."),
    "ours_supcond": ("#2a78d6", "-"), "ours_w05": ("#2a78d6", ":"), "ens_static": ("#4a3aa7", "-"),
    "ens_unc": ("#008300", "-"), "ens_er": ("#eda100", "-"), "ens_ecd": ("#e87ba4", "-"),
    "sup_ens_rand": ("#4a3aa7", "--"),
    "ours_div": ("#2a78d6", (0, (3, 1, 1, 1))), "ours_supcond_div": ("#eb6834", (0, (3, 1, 1, 1))),
}
plt.rcParams.update({"figure.constrained_layout.use": True, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight"})


DISPLAY = {"metabric": "METABRIC", "tcga_brca": "TCGA breast", "tcga_gbmlgg": "TCGA glioma",
           "metabric_semisynth": "METABRIC semi-synthetic", "synth_scm": "synthetic SCM",
           "uci_rnaseq": "UCI RNA-seq", "wdbc": "WDBC", "metabric1000": "METABRIC, 1,000 genes"}


def disp(d: str) -> str:
    return DISPLAY.get(d, d)


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("wrote", FIG / f"{name}.pdf")


def curves(ax, sub: pd.DataFrame, methods: list[str], title: str) -> None:
    for m in methods:
        s = sub[sub.method == m]
        if s.empty:
            continue
        c = np.stack(s.acc_curve.to_list())
        x = np.arange(c.shape[1])
        mu, sd = c.mean(0), c.std(0)
        col, ls = STYLE[m]
        ax.plot(x, mu, ls, color=col, lw=1.6, label=METHOD_LABELS[m])
        ax.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.12, lw=0)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("features acquired")
    ax.set_ylabel("test accuracy")


ALL_LEVELS = {"metabric": 1574, "metabric_semisynth": 1574, "synth_scm": 1500, "tcga_brca": 544,
             "uci_rnaseq": 601, "wdbc": 419}
CANONICAL_DATASETS = ["metabric", "tcga_brca", "metabric_semisynth", "synth_scm", "uci_rnaseq", "wdbc"]
MAIN_TEXT_DATASETS = ["metabric", "tcga_brca", "metabric_semisynth", "synth_scm"]


def fig_curves(main: pd.DataFrame, name_prefix: str = "curves", dataset_list=None) -> None:
    core = ["rand", "static_sup", "eig_sup_ens", "ens_rand", "ours", "ours_supcond"]
    dataset_list = dataset_list or CANONICAL_DATASETS
    main = main.copy()
    main["level"] = [("all" if n == ALL_LEVELS.get(d) else n) for d, n in zip(main.dataset, main.n_labels)]
    for n in sorted(main.level.unique(), key=lambda v: (isinstance(v, str), v)):
        dsets = [d for d in dataset_list
                 if not main[(main.dataset == d) & (main.level == n)].empty]
        if not dsets:
            continue
        fig, axes = plt.subplots(1, len(dsets), figsize=(3.2 * len(dsets), 2.7), squeeze=False)
        for i, (ax, d) in enumerate(zip(axes[0], dsets)):
            curves(ax, main[(main.dataset == d) & (main.level == n)], core, f"{disp(d)}, $N_l$={n}")
            if i > 0:
                ax.set_ylabel("")
        axes[0][0].legend(loc="lower right", fontsize=7)
        save(fig, f"{name_prefix}_n{n}")
    # every criterion on metabric at 50 labels
    sub = main[(main.dataset == "metabric") & (main.n_labels == 50)]
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        curves(ax, sub, ["rand", "static_sup", "unc", "eig_sup", "er_sup", "ecd_sup", "eig_sup_ens", "ssl_rand", "ens_rand", "ours_sslonly", "ours", "ours_supcond"],
               "METABRIC, $N_l$=50, all criteria")
        ax.legend(fontsize=6.5, ncol=2, loc="lower right")
        save(fig, "curves_metabric_n50_all")


def fig_label_efficiency(main: pd.DataFrame) -> None:
    for d in main.dataset.unique():
        sub = main[main.dataset == d]
        if sub.n_labels.nunique() < 3:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
        for m in ["rand", "static_sup", "eig_sup", "eig_sup_ens", "ens_rand", "ours_sslonly", "ours", "ours_supcond"]:
            g = sub[sub.method == m].groupby("n_labels")["aucc"].agg(["mean", "std"])
            if g.empty:
                continue
            col, ls = STYLE[m]
            axes[0].errorbar(g.index, g["mean"], yerr=g["std"], fmt="o" + ls, color=col, lw=1.4, ms=3.5,
                             capsize=2, label=METHOD_LABELS[m])
        axes[0].set_xscale("log")
        axes[0].set_xlabel("labeled rows $N_l$")
        axes[0].set_ylabel("AUCC (mean acc. over budgets 1..B)")
        axes[0].legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=1, framealpha=0.85, frameon=True)
        axes[0].set_title(f"{disp(d)}: label efficiency", fontsize=9)
        # gap ours - best supervised criterion per n
        piv = sub.pivot_table(index=["n_labels", "seed"], columns="method", values="aucc")
        sup = [c for c in ["eig_sup", "er_sup", "ecd_sup", "eig_sup_mfa", "eig_sup_ens"] if c in piv]
        best_by_n = piv.groupby(level=0)[sup].mean().idxmax(axis=1)
        head = "ours_supcond" if "ours_supcond" in piv and piv["ours_supcond"].notna().groupby(level=0).any().all() else "ours"
        gaps = []
        for n, s in piv.groupby(level=0):
            g = (s[head] - s[best_by_n[n]]).dropna()
            gaps.append((n, g.mean(), g.std()))
        gaps = np.array(gaps)
        axes[1].errorbar(gaps[:, 0], gaps[:, 1], yerr=gaps[:, 2], fmt="o-", color=STYLE[head][0], lw=1.4, ms=3.5, capsize=2)
        axes[1].axhline(0, color="k", lw=0.8)
        axes[1].set_xscale("log")
        axes[1].set_xlabel("labeled rows $N_l$")
        axes[1].set_ylabel(f"AUCC gap: {METHOD_LABELS[head]} $-$ best sup.")
        axes[1].set_title("gap versus label count", fontsize=9)
        save(fig, f"label_efficiency_{d}")


def fig_recovery(main: pd.DataFrame) -> None:
    dsets = [d for d in ["synth_scm", "metabric_semisynth"] if not main[main.dataset == d].empty]
    if not dsets or "precision_at_k" not in main:
        return
    methods = ["rand", "unc", "eig_sup", "eig_sup_ens", "ours_sslonly", "ours"]
    fig, axes = plt.subplots(len(dsets), 2, figsize=(7.0, 2.4 * len(dsets)), squeeze=False, sharey="row")
    for i, d in enumerate(dsets):
        sub = main[(main.dataset == d) & (main.n_labels == 50)]
        for j, (metric, lab) in enumerate([("precision_at_k", "precision@k of causal features"),
                                           ("redundancy_at_k", "redundant acquisitions@k")]):
            g = sub.groupby("method")[metric].agg(["mean", "std"]).reindex(methods).dropna()
            ax = axes[i][j]
            cols = [STYLE[m][0] for m in g.index]
            ax.barh(range(len(g)), g["mean"], xerr=g["std"], color=cols, height=0.6, capsize=2, alpha=0.9)
            ax.set_yticks(range(len(g)))
            if j == 0:
                ax.set_yticklabels([METHOD_LABELS[m] for m in g.index], fontsize=7.5)
            ax.set_xlabel(lab)
            ax.set_title(f"{disp(d)}, $N_l$=50", fontsize=9)
            ax.grid(axis="y", visible=False)
    save(fig, "recovery")


def fig_bars(df: pd.DataFrame, name: str, title: str) -> None:
    if df.empty:
        return
    for (d, n), sub in df.groupby(["dataset", "n_labels"]):
        g = sub.groupby("method")["aucc"].agg(["mean", "std"]).sort_values("mean")
        fig, ax = plt.subplots(figsize=(4.4, 0.4 * len(g) + 1.0))
        cols = [STYLE.get(m, ("#888888", "-"))[0] for m in g.index]
        ax.barh(range(len(g)), g["mean"], xerr=g["std"], color=cols, height=0.6, capsize=2)
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in g.index], fontsize=7.5)
        lo = max(0.0, g["mean"].min() - 3 * g["std"].max() - 0.02)
        ax.set_xlim(lo, min(1.0, g["mean"].max() + 3 * g["std"].max() + 0.02))
        ax.set_xlabel("AUCC")
        ax.set_title(f"{title}: {disp(d)}, $N_l$={n}", fontsize=9)
        ax.grid(axis="y", visible=False)
        save(fig, f"{name}_{d}_n{n}")


def fig_cost(cost: pd.DataFrame) -> None:
    if cost.empty:
        return
    for (d, n), sub in cost.groupby(["dataset", "n_labels"]):
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        budget = sub.cost_budget.iloc[0]
        grid = np.linspace(budget / 40, budget, 40)
        for m in ["rand", "eig_sup", "eig_sup_costblind", "ours", "ours_costblind"]:
            runs = sub[sub.method == m]
            if runs.empty:
                continue
            accs = []
            for rid in runs.run_id:
                z = np.load(Path("experiments") / rid / f"{m}.npz")

                class R:
                    pass
                r = R()
                r.cum_cost, r.correct = z["cum_cost"], z["correct"]
                accs.append(accuracy_at_cost_grid(r, grid))
            accs = np.stack(accs)
            col, ls = STYLE[m]
            ax.plot(grid, accs.mean(0), ls, color=col, lw=1.6, label=METHOD_LABELS[m])
            ax.fill_between(grid, accs.mean(0) - accs.std(0), accs.mean(0) + accs.std(0), color=col, alpha=0.12, lw=0)
        ax.set_xlabel("cumulative acquisition cost")
        ax.set_ylabel("test accuracy")
        ax.set_title(f"heterogeneous costs {{1,2,4}}: {disp(d)}, $N_l$={n}", fontsize=9)
        ax.legend(fontsize=6.5, loc="lower right")
        save(fig, f"cost_{d}_n{n}")


def fig_ssl_history() -> None:
    import json
    hist = None
    for p in sorted(Path("experiments").glob("main_metabric_s0_n50*/summary.json")):
        hist = json.load(open(p)).get("ssl_history")
    if not hist:
        return
    h = pd.DataFrame(hist)
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.plot(h.step, h.train_nll, color="#4a3aa7", lw=1.4, label="train NLL (masked)")
    ax.plot(h.step, h.val_nll, color="#2a78d6", lw=1.6, label="validation NLL")
    ax.set_xlabel("gradient step")
    ax.set_ylabel("Gaussian NLL per masked feature")
    ax.set_title("SSL pretraining, METABRIC, seed 0", fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "ssl_history_metabric")


def fig_eddi_dime(df: pd.DataFrame) -> None:
    """Grouped comparison: ours vs best supervised, eddi, eddi_ssl, dime, across label counts."""
    methods = ["eig_sup_ens", "eddi", "eddi_ssl", "dime", "ours", "ours_supcond"]
    labels = {"eig_sup_ens": "best supervised", "eddi": "EDDI", "eddi_ssl": "EDDI (pool)",
              "dime": "DIME", "ours": "SS-EIG", "ours_supcond": "ours"}
    colors = {"eig_sup_ens": "#4a3aa7", "eddi": "#e87ba4", "eddi_ssl": "#eda100",
             "dime": "#1baf7a", "ours": "#8fb8ea", "ours_supcond": "#2a78d6"}
    main = df[(df.experiment == "main") & (df.method.isin(methods))]
    for ds in ("metabric", "tcga_brca"):
        sub = main[(main.dataset == ds) & (main.seed < 5)]
        levels = sorted([n for n in sub.n_labels.unique() if n in (25, 50, 100, 200)])
        if not levels:
            continue
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        width = 0.15
        x = np.arange(len(levels))
        for i, m in enumerate(methods):
            means, stds = [], []
            for n in levels:
                v = sub[(sub.n_labels == n) & (sub.method == m)].aucc
                means.append(v.mean() if len(v) else np.nan)
                stds.append(v.std() if len(v) else 0)
            ax.bar(x + (i - 2) * width, means, width, yerr=stds, label=labels[m],
                  color=colors[m], capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in levels])
        ax.set_xlabel("labeled rows $N_l$")
        ax.set_ylabel("AUCC")
        ax.set_title(f"{disp(ds)}: published methods (5 seeds)", fontsize=9)
        ax.legend(fontsize=6.5, ncol=2, loc="lower right")
        lo = max(0.0, sub[sub.method.isin(methods)].aucc.min() - 0.05)
        ax.set_ylim(lo, min(1.0, sub[sub.method.isin(methods)].aucc.max() + 0.05))
        save(fig, f"eddi_dime_{ds}")


def fig_gap_cohorts(df: pd.DataFrame) -> None:
    """AUCC gap (proposed method - best supervised criterion, chosen per cell by mean) against labeled rows,
    for the three real cohorts, mean and 95% t-interval over seeds."""
    from scipy import stats as _stats

    sup = ["eig_sup", "er_sup", "ecd_sup", "eig_sup_ens"]
    cohorts = [("metabric", "METABRIC (breast, microarray)", "#2a78d6", "o", 0.955),
               ("tcga_brca", "TCGA breast (RNA-seq)", "#eb6834", "s", 1.0),
               ("tcga_gbmlgg", "TCGA glioma (RNA-seq)", "#1baf7a", "^", 1.045),
               ("uci_har", "Human activity (smartphone)", "#8a5cc2", "D", 1.09)]
    # filled marker: significant after correction within the cohort family (BH, as in the main table)
    sig = set()
    agg = Path("experiments/aggregate")
    if (agg / "stats_extra.json").exists():
        sig |= {(c["dataset"], int(c["n_labels"])) for c in json.load(open(agg / "stats_extra.json"))["ours_supcond"]["cells"]
                if c["significant_bh"]}
    if (agg / "gbmlgg_check.json").exists():
        sig |= {("tcga_gbmlgg", int(n)) for n in json.load(open(agg / "gbmlgg_check.json"))["significant_cells"]}
    if (agg / "har_check.json").exists():
        sig |= {("uci_har", int(n)) for n in json.load(open(agg / "har_check.json"))["significant_cells"]}
    main = df[df.experiment == "main"]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.axhline(0, color="#555555", lw=0.8)
    for ds, label, col, mk, dodge in cohorts:
        sub = main[main.dataset == ds]
        piv = sub.pivot_table(index=["n_labels", "seed"], columns="method", values="aucc")
        if "ours_supcond" not in piv:
            continue
        pts = []
        for n, s in piv.groupby(level=0):
            best = s[[c for c in sup if c in s]].mean().idxmax()
            g = (s["ours_supcond"] - s[best]).dropna()
            if len(g) < 2:
                continue
            half = _stats.t.ppf(0.975, len(g) - 1) * g.std(ddof=1) / np.sqrt(len(g))
            pts.append((n, g.mean(), half))
        pts = np.array(pts)
        ax.errorbar(pts[:, 0] * dodge, pts[:, 1], yerr=pts[:, 2], color=col, marker="none", lw=1.4,
                    capsize=2.5)
        filled = np.array([(ds, int(n)) in sig for n in pts[:, 0]])
        ax.scatter(pts[filled, 0] * dodge, pts[filled, 1], color=col, marker=mk, s=28, zorder=3, label=label)
        ax.scatter(pts[~filled, 0] * dodge, pts[~filled, 1], facecolor="white", edgecolor=col, marker=mk,
                   s=28, lw=1.2, zorder=3)
        dy = {"tcga_gbmlgg": 4, "uci_har": -8}.get(ds, 0)
        ax.annotate(label.split(" (")[0], (pts[-1, 0] * dodge, pts[-1, 1]), xytext=(6, dy), textcoords="offset points",
                    va="center", fontsize=7.5, color="#333333")
    ax.set_xscale("log")
    ax.set_xlabel("labeled rows $N_l$ (log scale)")
    ax.set_ylabel("AUCC gap: ours $-$ best supervised")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, loc="upper right")
    save(fig, "gap_cohorts")


def fig_label_type(df: pd.DataFrame) -> None:
    """AUCC gap (ours - best supervised) at N_l=50, grouped by whether the label tracks a
    broad expression program or a small number of specific genes."""
    import numpy as np
    from scipy import stats as _stats

    cells = [
        ("HER2 status", "metabric_her2", "specific"), ("ER status", "metabric_er", "specific"),
        ("semi-synthetic\n(6 genes)", "metabric_semisynth", "specific"),
        ("synthetic SCM\n(5 causal feats.)", "synth_scm", "specific"),
        ("integrative cluster", "metabric_intclust", "broad"),
        ("intrinsic subtype, METABRIC", "metabric", "broad"), ("histological grade", "metabric_grade", "broad"),
        ("intrinsic subtype, TCGA", "tcga_brca", "broad"), ("glioma expression cluster", "tcga_gbmlgg", "broad"),
    ]
    main = df[(df.experiment == "main") & (df.n_labels == 50)]
    rows = []
    for label, ds, kind in cells:
        sub = main[main.dataset == ds]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="seed", columns="method", values="aucc")
        sups = [c for c in ["eig_sup", "er_sup", "ecd_sup", "eig_sup_ens"] if c in piv]
        if not sups or "ours_supcond" not in piv:
            continue
        best = piv[sups].mean().idxmax()
        both = piv[["ours_supcond", best]].dropna()
        d = both["ours_supcond"] - both[best]
        p = _stats.ttest_rel(both["ours_supcond"], both[best]).pvalue if len(d) > 1 else float("nan")
        rows.append((label, kind, d.mean(), d.std(), p))
    rows.sort(key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    colors = {"broad": "#2a78d6", "specific": "#eb6834"}
    for i, (label, kind, mean, std, p) in enumerate(rows):
        ax.barh(i, mean, xerr=std, color=colors[kind], height=0.6, capsize=3)
        marker = "*" if p < 0.05 else ""
        ax.text(mean + (std + 0.004) * (1 if mean >= 0 else -1), i, marker, va="center",
               ha="left" if mean >= 0 else "right", fontsize=11, color="black")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("AUCC gap: ours $-$ best supervised ($N_l$=50)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in ("broad", "specific")]
    ax.legend(handles, ["clustering / multi-gene program", "single biomarker / few constructed features"], fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    ax.set_title("Advantage by label type ($^*$ = $p<0.05$, paired $t$-test)", fontsize=9)
    save(fig, "label_type_gap")


def main() -> None:
    df = load_runs()
    if df.empty:
        print("no runs")
        return
    main_df = df[df.experiment == "main"]
    if not main_df.empty:
        fig_curves(main_df[main_df.n_labels == 50], name_prefix="curves_main", dataset_list=MAIN_TEXT_DATASETS)
        fig_curves(main_df, name_prefix="curves", dataset_list=CANONICAL_DATASETS)
        fig_label_efficiency(main_df)
        fig_recovery(main_df)
        fig_ssl_history()
    fig_bars(df[df.experiment == "ablation"], "ablation", "ablation")
    fig_eddi_dime(df)
    fig_label_type(df)
    fig_gap_cohorts(df)
    fig_cost(df[df.experiment == "cost"])
    for exp in ("missing", "noise", "transfer", "capacity", "poolfrac", "objective"):
        fig_bars(df[df.experiment == exp], exp, exp)
    sc = df[df.experiment == "scale"]
    if not sc.empty:
        for n in sorted(sc.n_labels.unique()):
            fig, ax = plt.subplots(figsize=(4.2, 3.0))
            curves(ax, sc[sc.n_labels == n], ["rand", "static_sup", "eig_sup", "eig_sup_ens", "ens_rand", "ours", "ours_supcond"],
                   f"METABRIC, 1,000 genes, $N_l$={n}")
            ax.legend(fontsize=6.5, loc="lower right")
            save(fig, f"scale_metabric1000_n{n}")


if __name__ == "__main__":
    main()
