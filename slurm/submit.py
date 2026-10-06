#!/usr/bin/env python
"""Submit a queue: sbatch array if a cluster is available, otherwise the local runner."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def cluster_available() -> bool:
    if shutil.which("sbatch") is None:
        return False
    try:
        subprocess.run(["sinfo", "-h"], check=True, capture_output=True, timeout=10)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    ap.add_argument("--local", action="store_true", help="force the local runner")
    ap.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    n_jobs = sum(1 for l in open(args.queue) if l.strip())
    if cluster_available() and not args.local:
        print(f"[submit] cluster detected; sbatch array of {n_jobs} jobs")
        subprocess.run(["sbatch", f"--array=0-{n_jobs - 1}%4", "slurm/run_array.sbatch", args.queue], check=True)
        Path("logs").mkdir(exist_ok=True)
        with open("logs/submit.log", "a") as fh:
            fh.write(json.dumps({"queue": args.queue, "backend": "slurm", "n_jobs": n_jobs}) + "\n")
    else:
        print(f"[submit] no cluster; local runner for {n_jobs} jobs")
        with open("logs/submit.log", "a") as fh:
            fh.write(json.dumps({"queue": args.queue, "backend": "local", "n_jobs": n_jobs}) + "\n")
        subprocess.run([sys.executable, "slurm/local_runner.py", args.queue] + args.runner_args, check=False)


if __name__ == "__main__":
    main()
