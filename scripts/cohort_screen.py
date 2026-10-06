#!/usr/bin/env python
"""Screen for an independent non-breast cohort. Uses no acquisition method: for each
candidate TCGA cohort, primary tumors (barcode -01) in the Xena HiSeqV2 matrix, 200 top-variance
genes, and 5-fold stratified logistic regression (L2, C=1, standardized) on the expression-defined
PanCanAtlas subtype (Subtype_mRNA). Writes experiments/aggregate/cohort_screen.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CANDIDATES = {"KIRC": "KIRC", "THCA": "THCA", "OV": "OVCA", "GBMLGG": "GBM_LGG"}


def load(cohort: str, sub_name: str, sub: pd.DataFrame, n_features: int = 200):
    expr = pd.read_csv(f"data/raw/tcga_{cohort}/HiSeqV2.gz", sep="\t", index_col=0, compression="gzip")
    expr = expr.loc[:, [c for c in expr.columns if c.endswith("-01")]]
    expr = expr[~expr.index.duplicated(keep="first")].astype(np.float32)
    top = expr.var(axis=1).sort_values(ascending=False).index[:n_features]
    X = expr.loc[top].T
    lab = sub.loc[sub["cancer"] == sub_name, "Subtype_mRNA"].reindex(X.index)
    lab = lab.where(~lab.isin(["unclassified", "-"]))
    return X, lab


def main() -> None:
    sub = pd.read_csv("data/raw/tcga_subtypes/TCGASubtype.20170308.tsv.gz", sep="\t", index_col=0)
    sub["cancer"] = sub["Subtype_Selected"].astype(str).str.split(".").str[0]
    lines = ["| cohort | RNA-seq primaries | labeled | classes | LR accuracy (5-fold) | in [0.70, 0.90] |",
             "|---|---|---|---|---|---|"]
    for cohort, sub_name in CANDIDATES.items():
        X, lab = load(cohort, sub_name, sub)
        m = lab.notna().to_numpy()
        y = pd.factorize(lab[m])[0]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        acc = cross_val_score(clf, X.to_numpy()[m], y, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
        lines.append(f"| {cohort} | {len(X)} | {int(m.sum())} | {len(set(y))} | {acc:.3f} | "
                     f"{'yes' if 0.70 <= acc <= 0.90 else 'no'} |")
        print(lines[-1], flush=True)
    Path("experiments/aggregate/cohort_screen.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
