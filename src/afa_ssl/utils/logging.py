"""Structured JSONL log plus a human-readable mirror per run."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, log_root: str | Path, run_id: str) -> None:
        root = Path(log_root)
        root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._jsonl = open(root / f"run_{run_id}.jsonl", "a")
        self._text = open(root / f"run_{run_id}.log", "a")
        self.t0 = time.time()

    def event(self, kind: str, **fields: Any) -> None:
        rec = {"t": round(time.time() - self.t0, 3), "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "run_id": self.run_id, "kind": kind, **fields}
        self._jsonl.write(json.dumps(rec, default=_json_default) + "\n")
        self._jsonl.flush()
        msg = " ".join(f"{k}={_fmt(v)}" for k, v in fields.items())
        line = f"[{rec['ts']} +{rec['t']:8.1f}s] {kind:12s} {msg}"
        self._text.write(line + "\n")
        self._text.flush()
        print(line, flush=True)

    def close(self) -> None:
        self._jsonl.close()
        self._text.close()


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    s = str(v)
    return s if len(s) < 120 else s[:117] + "..."


def _json_default(o: Any):
    try:
        import numpy as np

        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)
