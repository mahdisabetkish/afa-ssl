#!/usr/bin/env python
"""Build and cache every processed dataset listed in configs/datasets/."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afa_ssl.utils.resources import configure_threads, limit_memory  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None, help="dataset names (default: all)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--data-root", default="data/processed")
    args = ap.parse_args()
    configure_threads(4)
    limit_memory()
    import numpy as np

    from afa_ssl.data.loaders import build_dataset
    from afa_ssl.utils.config import DataConfig, _from_dict, load_yaml

    files = sorted(Path("configs/datasets").glob("*.yaml"))
    for f in files:
        cfg = _from_dict(DataConfig, load_yaml(f))
        if args.datasets and cfg.name not in args.datasets:
            continue
        try:
            d = build_dataset(cfg, args.data_root, force=args.force)
        except FileNotFoundError as e:
            print(f"[skip] {cfg.name}: {e}")
            continue
        X, y = d["X"], d["y"]
        print(f"[ok] {cfg.name}: X={X.shape} classes={np.bincount(y[y >= 0]).tolist()} unlabeled={int((y < 0).sum())} "
              f"nan={int(np.isnan(X).sum())} bayes_acc={d.get('bayes_acc', 'n/a')}")


if __name__ == "__main__":
    main()
