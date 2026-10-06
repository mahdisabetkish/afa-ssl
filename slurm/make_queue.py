#!/usr/bin/env python
"""Write a JSONL job queue for an experiment grid.

    python slurm/make_queue.py --experiment main --datasets metabric:50,100,200,400,all \
        synth_scm:50,200,all --seeds 0 1 2 3 4 --out slurm/jobs/main.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, help="name of configs/experiments/<name>.yaml")
    ap.add_argument("--datasets", nargs="+", required=True, help="name:n1,n2,... ('all' = whole pool)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()
    lines = []
    for spec in args.datasets:
        name, levels = spec.split(":")
        for lvl in levels.split(","):
            n = 100000 if lvl == "all" else int(lvl)
            for seed in args.seeds:
                run_id = f"{args.experiment}_{name}_s{seed}_n{n}" + (f"_{args.tag}" if args.tag else "")
                cmd = ["scripts/run_experiment.py", "--experiment", f"configs/experiments/{args.experiment}.yaml",
                       "--dataset", f"configs/datasets/{name}.yaml", "--seed", str(seed), "--n-labels", str(n)]
                if args.tag:
                    cmd += ["--tag", args.tag]
                if args.set:
                    cmd += ["--set"] + args.set
                lines.append(json.dumps({"id": run_id, "cmd": cmd, "out_dir": f"experiments/{run_id}"}))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a" if args.append else "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} jobs to {args.out}")


if __name__ == "__main__":
    main()
