#!/usr/bin/env python
"""Selection screen for the non-omics dataset. Uses no acquisition method.

For each candidate, after the same preprocessing as every cohort (seeded cap at 2,000 rows, 200
top-variance features where D > 200), logistic regression (L2, C=1) on standardized features gives
full = 5-fold stratified CV accuracy with all labels, and few = mean accuracy over 10 draws of 50
stratified labeled rows, tested on all other rows. Label hunger = full - few; the candidate with the
largest label hunger is chosen. Writes experiments/aggregate/tabular_screen.{json,md}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afa_ssl.data.loaders import build_dataset  # noqa: E402
from afa_ssl.utils.config import DataConfig, load_yaml  # noqa: E402

CANDIDATES = ["uci_har", "uci_optdigits", "uci_satimage", "uci_mfeat"]
DATA_ROOT = "data/processed"


def screen(name: str) -> dict:
    cfg = DataConfig(**load_yaml(f"configs/datasets/{name}.yaml"))
    d = build_dataset(cfg, DATA_ROOT)
    X, y = np.asarray(d["X"], dtype=float), np.asarray(d["y"])
    lr = lambda: LogisticRegression(C=1.0, max_iter=5000)
    full = cross_val_score(make_pipeline(StandardScaler(), lr()), X, y,
                           cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
    Xs = StandardScaler().fit_transform(X)
    few = []
    for r in range(10):
        tr, te = train_test_split(np.arange(len(y)), train_size=50, stratify=y, random_state=r)
        few.append(lr().fit(Xs[tr], y[tr]).score(Xs[te], y[te]))
    return {"dataset": name, "rows": int(len(y)), "features": int(X.shape[1]),
            "classes": int(len(np.unique(y))), "full": float(full), "few": float(np.mean(few)),
            "label_hunger": float(full - np.mean(few))}


def main() -> None:
    rows = [screen(n) for n in CANDIDATES]
    chosen = max(rows, key=lambda r: r["label_hunger"])["dataset"]
    out = Path("experiments/aggregate")
    out.mkdir(parents=True, exist_ok=True)
    (out / "tabular_screen.json").write_text(json.dumps({"candidates": rows, "chosen": chosen}, indent=1))
    md = ["| dataset | rows | features | classes | LR full | LR 50 labels | label hunger |", "|" + "---|" * 7]
    md += [f"| {r['dataset']} | {r['rows']} | {r['features']} | {r['classes']} | {r['full']:.3f} | "
           f"{r['few']:.3f} | {r['label_hunger']:.3f} |" for r in rows]
    md.append(f"\nchosen: {chosen}")
    (out / "tabular_screen.md").write_text("\n".join(md) + "\n")
    names = {"uci_har": "Human activity (smartphone)", "uci_optdigits": "Handwritten digits",
             "uci_satimage": "Landsat satellite", "uci_mfeat": "Multiple features (digits)"}
    tex = ["\\begin{tabular}{lccccc}", "\\toprule",
           "dataset & features kept & classes & LR, all labels & LR, 50 labels & label hunger \\\\", "\\midrule"]
    tex += [f"{names[r['dataset']]}{' (chosen)' if r['dataset'] == chosen else ''} & {r['features']} & {r['classes']} & "
            f"{r['full']:.3f} & {r['few']:.3f} & {r['label_hunger']:.3f} \\\\" for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}"]
    Path("paper/tables/tabular_screen.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
