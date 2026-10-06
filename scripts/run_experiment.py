#!/usr/bin/env python
"""Run one (dataset, seed, n_labels) experiment.

    python scripts/run_experiment.py --experiment configs/experiments/smoke.yaml \
        --dataset configs/datasets/synth_scm.yaml --seed 0 --n-labels 50 [--set acq.budget=10]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afa_ssl.utils.resources import configure_threads, limit_memory  # noqa: E402


def parse_set(items: list[str]) -> dict:
    out: dict = {}
    for it in items:
        k, v = it.split("=", 1)
        try:
            import yaml

            v = yaml.safe_load(v)
        except Exception:
            pass
        cur = out
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-labels", type=int, default=50)
    ap.add_argument("--tag", default="")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-ram-gb", type=float, default=6.0)
    ap.add_argument("--add-methods", default=None,
                    help="comma-separated methods to add to the existing run directory (merge mode)")
    ap.add_argument("--status-file", default=None,
                    help="explicit status filename (default: status.json, or status_add.json in merge mode); "
                         "queues that merge different method sets into the same run directory must each "
                         "pass a distinct name so the local runner's resume check cannot alias them")
    args = ap.parse_args()

    configure_threads(args.threads)
    limit_memory(int(args.max_ram_gb * 1024**3))

    from afa_ssl.experiment import Run
    from afa_ssl.utils.config import build_run_config

    over = parse_set(args.set)
    over.update({"seed": args.seed, "n_labels": args.n_labels, "tag": args.tag, "threads": args.threads,
                 "max_ram_gb": args.max_ram_gb})
    cfg = build_run_config(args.experiment, args.dataset, over)
    if args.add_methods:
        cfg.acq.methods = [m.strip() for m in args.add_methods.split(",") if m.strip()]
        Run(cfg, merge=True, status_file=args.status_file).execute()
    else:
        Run(cfg, status_file=args.status_file).execute()


if __name__ == "__main__":
    main()
