"""Small helpers for run directories, the registry and git metadata."""
from __future__ import annotations

import csv
import fcntl
import json
import subprocess
from pathlib import Path
from typing import Any

REGISTRY_COLUMNS = [
    "run_id", "git_commit", "config_hash", "dataset", "method", "seed", "n_labels",
    "aucc", "acc_final", "acc_full", "cost_budget", "status", "runtime_s", "timestamp",
]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def append_registry(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        w = csv.DictWriter(fh, fieldnames=REGISTRY_COLUMNS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in REGISTRY_COLUMNS})
        fcntl.flock(fh, fcntl.LOCK_UN)


def write_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)


def read_json(path: str | Path) -> Any:
    with open(path) as fh:
        return json.load(fh)
