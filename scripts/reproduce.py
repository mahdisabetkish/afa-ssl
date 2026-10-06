#!/usr/bin/env python
"""Regenerate every number in the paper from scratch:
data -> tests -> queues -> runs (local runner, resumable) -> aggregate -> figures.

    python scripts/reproduce.py [--skip-runs] [--seeds 0 1 2 3 4]
"""
from __future__ import annotations

import argparse
import subprocess
import sys

PY = sys.executable
GRID = {
    "main": ["metabric:50,100,200,400,all", "metabric_semisynth:50,200,all", "synth_scm:50,200,all",
             "uci_rnaseq:50,200,all", "wdbc:25,100,all"],
    "ablation": ["metabric:50,200", "metabric_semisynth:50"],
    "labels": ["metabric_er:50,200", "metabric_her2:50,200", "metabric_intclust:50,200", "metabric_grade:50,200"],
    "scale": ["metabric1000:50,200"],
    "cost": ["metabric:50"],
}
# secondary experiments: (experiment name, datasets, extra --set overrides, tag)
SECONDARY = [
    ("missing", ["metabric:50"], ["data.pool_missing_rate=0.3", "acq.methods=[rand,eig_sup,eig_sup_ens,ens_rand,ours]"], ""),
    ("noise", ["metabric:50"], ["data.test_noise_sd=0.5", "acq.methods=[rand,eig_sup,eig_sup_ens,ens_rand,ours]"], ""),
    ("transfer", ["metabric:50"], ["data.pretrain_cohorts=[1]", "data.eval_cohorts=[2,3,4,5]",
                                   "acq.methods=[rand,eig_sup,eig_sup_ens,ens_rand,ours]"], ""),
    ("capacity", ["metabric:50"], ["model.hidden=64", "model.latent=32", "acq.methods=[ens_rand,ours]"], "small"),
    ("capacity", ["metabric:50"], ["model.hidden=256", "model.latent=128", "acq.methods=[ens_rand,ours]"], "large"),
    ("poolfrac", ["metabric:50"], ["train.ssl_pool_frac=0.1", "acq.methods=[ens_rand,ours]"], "f0.1"),
    ("poolfrac", ["metabric:50"], ["train.ssl_pool_frac=0.3", "acq.methods=[ens_rand,ours]"], "f0.3"),
    ("objective", ["metabric:50"], ["model.ssl_objective=mse", "acq.methods=[ens_rand,ours]"], "mse"),
    ("objective", ["metabric:50"], ["mask.budget_mix=0.0", "acq.methods=[ens_rand,ours]"], "unifmask"),
]


def sh(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-runs", action="store_true")
    ap.add_argument("--seeds", nargs="+", default=["0", "1", "2", "3", "4"])
    args = ap.parse_args()
    sh([PY, "scripts/make_data.py"])
    sh([PY, "-m", "pytest", "-q"])
    if not args.skip_runs:
        for exp, dsets in GRID.items():
            q = f"slurm/jobs/{exp}.jsonl"
            sh([PY, "slurm/make_queue.py", "--experiment", exp, "--datasets", *dsets, "--seeds", *args.seeds, "--out", q])
            sh([PY, "slurm/submit.py", q])
        q = "slurm/jobs/secondary.jsonl"
        first = True
        for exp, dsets, sets, tag in SECONDARY:
            cmd = [PY, "slurm/make_queue.py", "--experiment", exp, "--datasets", *dsets, "--seeds", *args.seeds,
                   "--out", q, "--set", *sets]
            if tag:
                cmd += ["--tag", tag]
            if not first:
                cmd.append("--append")
            first = False
            sh(cmd)
        sh([PY, "slurm/submit.py", q])
    sh([PY, "scripts/aggregate.py"])
    sh([PY, "scripts/make_figures.py"])


if __name__ == "__main__":
    main()
