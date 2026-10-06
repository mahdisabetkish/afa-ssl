#!/usr/bin/env python
"""Build a merge-mode queue: add methods to existing main-grid run directories
without repeating any computation. Each merge purpose gets a status filename
derived from its own method list, so two different merges can never be mistaken
for each other.

    python slurm/make_merge_queue.py --add-methods eddi,eddi_ssl,dime \
        --datasets metabric:25,50,100,200 tcga_brca:25,50,100,200 --seeds 0 1 2 3 4 \
        --out slurm/jobs/baseline_compare.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--add-methods", required=True, help="comma-separated methods to add")
    ap.add_argument("--datasets", nargs="+", required=True, help="name:n1,n2,... ('all' = whole pool)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--experiment", default="main", help="name of configs/experiments/<name>.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    methods = sorted(m.strip() for m in args.add_methods.split(",") if m.strip())
    # the status filename is derived from the method set itself: two queues that add the same
    # methods share a status file (so they dedupe correctly across reruns); two queues that add
    # different methods can never alias each other's status
    tag = hashlib.sha1(",".join(methods).encode()).hexdigest()[:8]
    status_name = f"status_merge_{tag}.json"

    lines = []
    for spec in args.datasets:
        name, levels = spec.split(":")
        for lvl in levels.split(","):
            n = 100000 if lvl == "all" else int(lvl)
            for seed in args.seeds:
                rid = f"{args.experiment}_{name}_s{seed}_n{n}"
                out_dir = f"experiments/{rid}"
                cmd = ["scripts/run_experiment.py", "--experiment", f"configs/experiments/{args.experiment}.yaml",
                       "--dataset", f"configs/datasets/{name}.yaml", "--seed", str(seed), "--n-labels", str(n),
                       "--add-methods", ",".join(methods), "--status-file", status_name]
                if args.set:
                    cmd += ["--set"] + args.set
                lines.append(json.dumps({"id": f"merge{tag}_{rid}", "out_dir": out_dir,
                                         "status_file": f"{out_dir}/{status_name}", "cmd": cmd}))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} jobs to {args.out} (methods={methods}, status_file={status_name})")


if __name__ == "__main__":
    main()
