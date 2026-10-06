#!/usr/bin/env python
"""How much does the acquisition ranking change when the predictor is swapped, compared
with when the feature model is swapped?

For freshly trained instances (METABRIC and TCGA breast, N_l in {25, 50}, seeds 0-2, main-grid
configuration) and 100 test rows with random observed sets of 5, 10 and 20 features, the EIG of
every unobserved candidate is computed with the method's 3-node rule under four combinations:
predictor in {supervised, proposed mixture} x feature model in {labeled-row Gaussian, pretrained
decoder}. Reported: median Spearman correlation, top-5 overlap and top-1 agreement between rankings
for a predictor swap (feature model fixed) and a feature-model swap (predictor fixed), and, as a
descriptive check, the per-instance ranking change against the AUCC gain of the proposed method over
supervised EIG in the matching main-grid run. Nothing under experiments/<run_id>/ is written.
Writes experiments/aggregate/ranking_stability.{json,md} and paper/tables/ranking_stability.tex.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from afa_ssl.utils.resources import configure_threads  # noqa: E402

configure_threads(1)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from afa_ssl.acquisition.criteria import gauss_hermite, scores_from_probs  # noqa: E402
from afa_ssl.experiment import Run  # noqa: E402
from afa_ssl.utils.config import build_run_config  # noqa: E402
from afa_ssl.utils.seed import seed_everything  # noqa: E402

N_ROWS, K_LEVELS, CHUNK = 100, (5, 10, 20), 10
COHORTS, LEVELS, SEEDS = ("metabric", "tcga_brca"), (25, 50), (0, 1, 2)


def eig(clf, cond, x, m, t, w):
    mu, sd = cond.conditional(x, m)
    vals = mu[..., None] + sd[..., None] * t[None, None, :]
    with torch.no_grad():
        p = F.softmax(clf.forward_candidates(x, m, vals), dim=-1)
        p0 = F.softmax(clf(x, m), dim=-1)
    return scores_from_probs(p, w, p0, "eig")


def compare(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    top = lambda v, k: set(np.argsort(-v)[:k].tolist())
    return (float(spearmanr(a, b).correlation), len(top(a, 5) & top(b, 5)) / 5.0,
            float(np.argmax(a) == np.argmax(b)))


def instance(ds: str, n: int, seed: int, tmp: Path) -> dict:
    over = {"seed": seed, "n_labels": n, "tag": "ranking_check", "threads": 1, "max_ram_gb": 6.0,
            "out_root": str(tmp), "log_root": str(tmp)}
    cfg = build_run_config(str(ROOT / "configs/experiments/main.yaml"), str(ROOT / f"configs/datasets/{ds}.yaml"), over)
    run = Run(cfg)
    run.rng = seed_everything(cfg.seed)
    run.prepare_data()
    preds = {"sup": run.get("sup"), "ours": run.get("ens")}
    conds = {"gauss": run.conditional_for("gauss_lab"), "decoder": run.conditional_for("mfa_pool")}
    t, w = (torch.from_numpy(a) for a in gauss_hermite(3))
    X = torch.from_numpy(run.X_test[:N_ROWS])
    n_rows, d = X.shape
    rng = np.random.default_rng(12345)
    rec = {"pred_swap": [], "cond_swap": []}
    for k in K_LEVELS:
        m = torch.zeros(n_rows, d)
        for i in range(n_rows):
            m[i, rng.choice(d, size=k, replace=False)] = 1.0
        for s in range(0, n_rows, CHUNK):
            x, mm = X[s:s + CHUNK], m[s:s + CHUNK]
            sc = {(p, c): eig(preds[p], conds[c], x, mm, t, w) for p in preds for c in conds}
            for i in range(len(x)):
                free = mm[i] < 0.5
                v = {key: val[i, free].numpy() for key, val in sc.items()}
                for c in conds:
                    rec["pred_swap"].append(compare(v[("sup", c)], v[("ours", c)]))
                for p in preds:
                    rec["cond_swap"].append(compare(v[(p, "gauss")], v[(p, "decoder")]))
    summ = {kind: {"rho": float(np.median([r[0] for r in rows])), "top5": float(np.mean([r[1] for r in rows])),
                   "top1": float(np.mean([r[2] for r in rows]))} for kind, rows in rec.items()}
    main = json.load(open(ROOT / "experiments" / f"main_{ds}_s{seed}_n{n}" / "summary.json"))["methods"]
    summ["aucc_gain_vs_eig_sup"] = main["ours_supcond"]["aucc"] - main["eig_sup"]["aucc"]
    return summ


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="ranking_check_"))
    res = {}
    try:
        for ds in COHORTS:
            for n in LEVELS:
                for seed in SEEDS:
                    res[f"{ds}/n{n}/s{seed}"] = instance(ds, n, seed, tmp)
                    print(f"{ds} n{n} s{seed}: {res[f'{ds}/n{n}/s{seed}']}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    agg = ROOT / "experiments" / "aggregate"
    (agg / "ranking_stability.json").write_text(json.dumps(res, indent=1))
    md = ["| cohort | N_l | predictor swap: rho / top-5 / top-1 | feature-model swap: rho / top-5 / top-1 |",
          "|---|---|---|---|"]
    tex = ["\\begin{tabular}{llcccccc}", "\\toprule",
           " & & \\multicolumn{3}{c}{swap the predictor} & \\multicolumn{3}{c}{swap the feature model} \\\\",
           "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
           "cohort & $\\Nl$ & $\\rho$ & top-5 & top-1 & $\\rho$ & top-5 & top-1 \\\\", "\\midrule"]
    for ds in COHORTS:
        for n in LEVELS:
            rows = [res[f"{ds}/n{n}/s{s}"] for s in SEEDS]
            g = lambda kind, key: float(np.mean([r[kind][key] for r in rows]))
            name = {"metabric": "METABRIC", "tcga_brca": "TCGA"}[ds]
            md.append(f"| {name} | {n} | {g('pred_swap','rho'):.3f} / {g('pred_swap','top5'):.2f} / {g('pred_swap','top1'):.2f} | "
                      f"{g('cond_swap','rho'):.3f} / {g('cond_swap','top5'):.2f} / {g('cond_swap','top1'):.2f} |")
            tex.append(f"{name} & {n} & {g('pred_swap','rho'):.3f} & {g('pred_swap','top5'):.2f} & {g('pred_swap','top1'):.2f} & "
                       f"{g('cond_swap','rho'):.3f} & {g('cond_swap','top5'):.2f} & {g('cond_swap','top1'):.2f} \\\\")
    change = [1 - r["pred_swap"]["rho"] for r in res.values()]
    gain = [r["aucc_gain_vs_eig_sup"] for r in res.values()]
    rho_link = spearmanr(change, gain)
    md.append("")
    md.append(f"Descriptive: Spearman(1 - rho_predictor_swap, AUCC gain of ours over eig_sup) across "
              f"{len(change)} instances = {rho_link.correlation:.3f} (p={rho_link.pvalue:.3f}).")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (agg / "ranking_stability.md").write_text("\n".join(md) + "\n")
    (ROOT / "paper" / "tables" / "ranking_stability.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
