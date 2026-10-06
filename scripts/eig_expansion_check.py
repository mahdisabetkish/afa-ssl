#!/usr/bin/env python
"""Numerical check of Proposition 1, prop:expansion (second-order expansion of the information-gain score).

For one freshly trained instance of the proposed predictor per cohort (seed 0, N_l=50, the
main-grid configuration) and random partial records drawn from the test rows, compare, over
the unobserved candidates of each record:

  ref    EIG under the conditional q = N(mu, sigma^2), 40-node Gauss-Hermite rule
  gh3    the same with the 3-node rule the method uses
  approx the second-order term 0.5 * sigma^2 * J(mu), J = sum_y p_y (d/dx_f log p_y)^2
  unif   EIG under the uniform distribution with the same mean and variance
  twopt  EIG under the two-point distribution mu +- sigma (same mean and variance)

for both conditional models (the Gaussian of the proposed method and the pretrained decoder
of SS-EIG). Writes experiments/aggregate/eig_expansion_check.{json,md} and
paper/tables/eig_expansion.tex. Nothing under experiments/<run_id>/ is touched: the run's
config and log go to a temporary directory.

    python scripts/eig_expansion_check.py
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
from numpy.polynomial.legendre import leggauss  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from afa_ssl.acquisition.criteria import gauss_hermite, scores_from_probs  # noqa: E402
from afa_ssl.experiment import Run  # noqa: E402
from afa_ssl.utils.config import build_run_config  # noqa: E402
from afa_ssl.utils.seed import seed_everything  # noqa: E402

N_ROWS = 100
K_LEVELS = (0, 5, 10, 20)
CHUNK = 10
H = 1e-2  # central-difference step for d log p / dx_f (standardized units)


def entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * torch.log(p.clamp_min(1e-12))).sum(-1)


def eig_rule(clf, x, m, mu, sd, t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """EIG of every candidate when x_f takes values mu + sd * t_j with probabilities w_j."""
    vals = mu[..., None] + sd[..., None] * t[None, None, :]
    with torch.no_grad():
        p = F.softmax(clf.forward_candidates(x, m, vals), dim=-1)          # (N, D, J, C)
    pbar = (w[None, None, :, None] * p).sum(2)
    return entropy(pbar) - (w[None, None, :] * entropy(p)).sum(2)


def second_order(clf, x, m, mu, sd) -> torch.Tensor:
    vals = mu[..., None] + torch.tensor([-H, 0.0, H])[None, None, :]
    with torch.no_grad():
        lp = F.log_softmax(clf.forward_candidates(x, m, vals), dim=-1)    # (N, D, 3, C)
    grad = (lp[:, :, 2] - lp[:, :, 0]) / (2 * H)
    fisher = (lp[:, :, 1].exp() * grad**2).sum(-1)
    return 0.5 * sd**2 * fisher


def check(ds: str, tmp: Path) -> dict:
    over = {"seed": 0, "n_labels": 50, "tag": "expansion_check", "threads": 1, "max_ram_gb": 6.0,
            "out_root": str(tmp), "log_root": str(tmp)}
    cfg = build_run_config(str(ROOT / "configs/experiments/main.yaml"),
                           str(ROOT / f"configs/datasets/{ds}.yaml"), over)
    run = Run(cfg)
    run.rng = seed_everything(cfg.seed)
    run.prepare_data()
    clf = run.get("ens")
    conds = {"gaussian": run.conditional_for("gauss_lab"), "decoder": run.conditional_for("mfa_pool")}

    gh40_t, gh40_w = (torch.from_numpy(a) for a in gauss_hermite(40))
    gh3_t, gh3_w = (torch.from_numpy(a) for a in gauss_hermite(3))
    u, uw = leggauss(20)
    un_t, un_w = torch.tensor(np.sqrt(3.0) * u, dtype=torch.float32), torch.tensor(uw / 2.0, dtype=torch.float32)
    tp_t, tp_w = torch.tensor([-1.0, 1.0]), torch.tensor([0.5, 0.5])

    X = torch.from_numpy(run.X_test[:N_ROWS])
    n, d = X.shape
    rng = np.random.default_rng(12345)
    out: dict = {}
    for cname, cond in conds.items():
        rows = []
        for k in K_LEVELS:
            m = torch.zeros(n, d)
            for i in range(n):
                if k:
                    m[i, rng.choice(d, size=k, replace=False)] = 1.0
            for s in range(0, n, CHUNK):
                x, mm = X[s:s + CHUNK], m[s:s + CHUNK]
                mu, sd = cond.conditional(x, mm)
                ref = eig_rule(clf, x, mm, mu, sd, gh40_t, gh40_w)
                gh3 = eig_rule(clf, x, mm, mu, sd, gh3_t, gh3_w)
                if s == 0:   # the 3-node score must equal the method's own implementation
                    vals = mu[..., None] + sd[..., None] * gh3_t[None, None, :]
                    with torch.no_grad():
                        p = F.softmax(clf.forward_candidates(x, mm, vals), dim=-1)
                        p0 = F.softmax(clf(x, mm), dim=-1)
                    own = scores_from_probs(p, gh3_w, p0, "eig")
                    assert torch.allclose(own, gh3, atol=1e-5), "3-node EIG differs from the method's"
                approx = second_order(clf, x, mm, mu, sd)
                unif = eig_rule(clf, x, mm, mu, sd, un_t, un_w)
                twopt = eig_rule(clf, x, mm, mu, sd, tp_t, tp_w)
                for i in range(len(x)):
                    free = mm[i] < 0.5
                    r = ref[i, free].numpy()
                    row = {"k": k, "median_sigma": float(sd[i, free].median())}
                    for name, a in (("approx", approx), ("gh3", gh3), ("unif", unif), ("twopt", twopt)):
                        v = a[i, free].numpy()
                        row[f"rho_{name}"] = float(spearmanr(r, v).correlation)
                        row[f"top1_{name}"] = float(np.argmax(v) == np.argmax(r))
                        row[f"relerr_{name}"] = float(np.median(np.abs(v - r) / np.maximum(r, 1e-8)))
                    rows.append(row)
        out[cname] = rows
    return out


def summarise(res: dict) -> tuple[dict, list[str], list[str]]:
    summ, md = {}, ["| cohort | conditional | k | median sigma | rho approx | top-1 approx | rel. err approx | "
                    "rho 3-node | rho uniform | rho two-point |", "|" + "---|" * 10]
    tex = ["\\begin{tabular}{llrcccccc}", "\\toprule",
           "cohort & feature model & $|O|$ & median $\\sigma$ & $\\rho$ (2nd order) & top-1 (2nd order) & "
           "$\\rho$ (3 nodes) & $\\rho$ (uniform) & $\\rho$ (two-point) \\\\", "\\midrule"]
    for ds, byc in res.items():
        for cname, rows in byc.items():
            for k in K_LEVELS:
                rr = [r for r in rows if r["k"] == k]
                med = lambda key: float(np.nanmedian([r[key] for r in rr]))
                mean = lambda key: float(np.mean([r[key] for r in rr]))
                s = {"median_sigma": med("median_sigma"), "rho_approx": med("rho_approx"),
                     "top1_approx": mean("top1_approx"), "relerr_approx": med("relerr_approx"),
                     "rho_gh3": med("rho_gh3"), "rho_unif": med("rho_unif"), "rho_twopt": med("rho_twopt"),
                     "top1_twopt": mean("top1_twopt"), "top1_unif": mean("top1_unif"), "n_records": len(rr)}
                summ[f"{ds}/{cname}/k{k}"] = s
                name = {"metabric": "METABRIC", "tcga_brca": "TCGA"}[ds]
                md.append(f"| {name} | {cname} | {k} | {s['median_sigma']:.2f} | {s['rho_approx']:.3f} | "
                          f"{s['top1_approx']:.2f} | {s['relerr_approx']:.3f} | {s['rho_gh3']:.3f} | "
                          f"{s['rho_unif']:.3f} | {s['rho_twopt']:.3f} |")
                tex.append(f"{name} & {cname} & {k} & {s['median_sigma']:.2f} & {s['rho_approx']:.3f} & "
                           f"{s['top1_approx']:.2f} & {s['rho_gh3']:.3f} & {s['rho_unif']:.3f} & "
                           f"{s['rho_twopt']:.3f} \\\\")
            tex.append("\\addlinespace")
    tex = tex[:-1] + ["\\bottomrule", "\\end{tabular}"]
    return summ, md, tex


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="expansion_check_"))
    try:
        res = {ds: check(ds, tmp) for ds in ("metabric", "tcga_brca")}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    summ, md, tex = summarise(res)
    agg = ROOT / "experiments" / "aggregate"
    (agg / "eig_expansion_check.json").write_text(json.dumps({"summary": summ, "records": res}, indent=1))
    (agg / "eig_expansion_check.md").write_text("\n".join(md) + "\n")
    (ROOT / "paper" / "tables" / "eig_expansion.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
