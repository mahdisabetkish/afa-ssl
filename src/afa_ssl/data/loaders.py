"""Dataset builders. Each returns a dict with X (float32), y (int64), names and
ground-truth metadata when available, and caches the result as .npz + meta.json."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from afa_ssl.data.synthetic import make_semisynth, make_synth_scm
from afa_ssl.utils.config import DataConfig


def top_variance_columns_chunked(reader, n_keep: int, transpose: bool) -> np.ndarray:
    """One pass over a chunked reader accumulating per-feature mean and variance.
    If `transpose`, rows of the file are features (METABRIC layout)."""
    if transpose:
        variances = []
        for chunk in reader:
            vals = chunk.to_numpy(dtype=np.float64)
            variances.append(np.nanvar(vals, axis=1))
        v = np.concatenate(variances)
    else:
        n = 0
        s = None
        s2 = None
        for chunk in reader:
            vals = chunk.to_numpy(dtype=np.float64)
            if s is None:
                s = np.zeros(vals.shape[1])
                s2 = np.zeros(vals.shape[1])
            s += np.nansum(vals, axis=0)
            s2 += np.nansum(vals**2, axis=0)
            n += vals.shape[0]
        v = s2 / n - (s / n) ** 2
    v = np.nan_to_num(v, nan=-1.0)
    return np.argsort(-v)[:n_keep]


def load_metabric(cfg: DataConfig) -> dict:
    raw = Path(cfg.raw_dir)
    expr_path = raw / "data_mrna_illumina_microarray.txt"
    # pass 1: variance per gene (genes are rows)
    reader = pd.read_csv(expr_path, sep="\t", chunksize=2000, index_col=0,
                         usecols=lambda c: c != "Entrez_Gene_Id", low_memory=False)
    keep = top_variance_columns_chunked(reader, cfg.n_features, transpose=True)
    keep_set = set(keep.tolist())
    # pass 2: read only those rows
    rows = []
    row_idx = 0
    for chunk in pd.read_csv(expr_path, sep="\t", chunksize=2000, index_col=0,
                             usecols=lambda c: c != "Entrez_Gene_Id", low_memory=False):
        sel = [i for i in range(len(chunk)) if row_idx + i in keep_set]
        rows.append(chunk.iloc[sel])
        row_idx += len(chunk)
    expr = pd.concat(rows)
    expr = expr[~expr.index.duplicated(keep="first")]
    X = expr.T  # samples x genes
    X.index.name = "SAMPLE_ID"
    clin = pd.read_csv(raw / "data_clinical_patient.txt", sep="\t", comment="#")
    clin = clin.set_index("PATIENT_ID")
    label_col = cfg.label_col or "CLAUDIN_SUBTYPE"
    if cfg.label_file == "sample":
        # sample ids coincide with patient ids in METABRIC (checked: 0 mismatches)
        samp = pd.read_csv(raw / "data_clinical_sample.txt", sep="\t", comment="#").set_index("SAMPLE_ID")
        lab = samp[label_col].reindex(X.index)
    else:
        lab = clin[label_col].reindex(X.index)
    cohort = clin["COHORT"].reindex(X.index)
    ok = lab.notna() & (lab.astype(str) != "NC") & (lab.astype(str) != "NA") & (lab.astype(str).str.strip() != "")
    X = X[ok]
    lab = lab[ok].astype(str).str.replace(r"\.0$", "", regex=True)
    cohort = cohort[ok].fillna(-1).astype(int).to_numpy()
    classes = sorted(lab.unique())
    y = np.array([classes.index(v) for v in lab], dtype=np.int64)
    Xv = X.to_numpy(dtype=np.float32)
    # a handful of NaNs may remain; fill with the column median
    if np.isnan(Xv).any():
        med = np.nanmedian(Xv, axis=0)
        Xv = np.where(np.isnan(Xv), med[None, :], Xv)
    out = {"X": Xv, "y": y, "feature_names": list(X.columns), "class_names": classes,
           "sample_ids": list(X.index), "cohort": cohort}
    return out


def load_uci_rnaseq(cfg: DataConfig) -> dict:
    zpath = Path(cfg.raw_dir) / "uci_rnaseq.zip"
    with zipfile.ZipFile(zpath) as z:
        inner = [n for n in z.namelist() if n.endswith(".tar.gz")]
        if inner:
            import tarfile

            with z.open(inner[0]) as fh:
                data = io.BytesIO(fh.read())
            with tarfile.open(fileobj=data, mode="r:gz") as tf:
                members = {m.name.split("/")[-1]: m for m in tf.getmembers() if m.isfile()}
                data_bytes = tf.extractfile(members["data.csv"]).read()
                label_bytes = tf.extractfile(members["labels.csv"]).read()
        else:
            data_bytes = z.read([n for n in z.namelist() if n.endswith("data.csv")][0])
            label_bytes = z.read([n for n in z.namelist() if n.endswith("labels.csv")][0])
    reader = pd.read_csv(io.BytesIO(data_bytes), chunksize=100, index_col=0)
    keep = top_variance_columns_chunked(reader, cfg.n_features, transpose=False)
    header = pd.read_csv(io.BytesIO(data_bytes), nrows=0, index_col=0).columns
    cols = [header[i] for i in keep]
    df = pd.read_csv(io.BytesIO(data_bytes), index_col=0, usecols=["Unnamed: 0"] + cols)
    df = df[cols]
    labels = pd.read_csv(io.BytesIO(label_bytes), index_col=0).reindex(df.index)["Class"]
    classes = sorted(labels.unique())
    y = np.array([classes.index(v) for v in labels], dtype=np.int64)
    return {"X": df.to_numpy(dtype=np.float32), "y": y, "feature_names": cols,
            "class_names": classes, "sample_ids": list(df.index)}


def load_tcga_brca(cfg: DataConfig) -> dict:
    """TCGA BRCA RNA-seq (UCSC Xena HiSeqV2, log2(norm_count + 1)) with PAM50 calls.
    Primary tumours only; tumours without a PAM50 call get y = -1 (unlabeled pool rows)."""
    raw = Path(cfg.raw_dir)
    expr_path = raw / "HiSeqV2.gz"
    clin = pd.read_csv(raw / "BRCA_clinicalMatrix", sep="\t", index_col=0, low_memory=False)
    primary = clin.index[clin["sample_type"].astype(str) == "Primary Tumor"]
    header = pd.read_csv(expr_path, sep="\t", nrows=0, compression="gzip").columns[1:]
    keep_cols = [c for c in header if c in set(primary)]
    # pass 1: per-gene variance over the kept samples (genes are rows)
    reader = pd.read_csv(expr_path, sep="\t", chunksize=2000, index_col=0, compression="gzip",
                         usecols=["sample"] + keep_cols)
    keep = top_variance_columns_chunked(reader, cfg.n_features, transpose=True)
    keep_set = set(keep.tolist())
    rows, row_idx = [], 0
    for chunk in pd.read_csv(expr_path, sep="\t", chunksize=2000, index_col=0, compression="gzip",
                             usecols=["sample"] + keep_cols):
        sel = [i for i in range(len(chunk)) if row_idx + i in keep_set]
        rows.append(chunk.iloc[sel])
        row_idx += len(chunk)
    expr = pd.concat(rows)
    expr = expr[~expr.index.duplicated(keep="first")]
    X = expr.T
    label_col = cfg.label_col or "PAM50Call_RNAseq"
    lab = clin[label_col].reindex(X.index).astype(str).str.strip()
    lab = lab.where(~lab.isin(["nan", "", "NA"]), other="")
    classes = sorted(v for v in lab.unique() if v != "")
    y = np.array([classes.index(v) if v != "" else -1 for v in lab], dtype=np.int64)
    Xv = X.to_numpy(dtype=np.float32)
    if np.isnan(Xv).any():
        med = np.nanmedian(Xv, axis=0)
        Xv = np.where(np.isnan(Xv), med[None, :], Xv)
    return {"X": Xv, "y": y, "feature_names": list(X.columns), "class_names": classes,
            "sample_ids": list(X.index), "n_unlabeled": int((y < 0).sum())}


def load_tcga_subtype(cfg: DataConfig) -> dict:
    """Any TCGA cohort from UCSC Xena (HiSeqV2, log2(norm_count + 1)), labeled by the PanCanAtlas
    subtype table (cfg.label_file, column cfg.label_col, rows of cancer cfg.subtype_cancer).
    Primary tumours (barcode -01) only; same gene selection and imputation as load_tcga_brca.
    Samples without a label, or labeled 'unclassified' / '-', get y = -1 (unlabeled pool rows)."""
    expr = pd.read_csv(Path(cfg.raw_dir) / "HiSeqV2.gz", sep="\t", index_col=0, compression="gzip")
    expr = expr.loc[:, [c for c in expr.columns if str(c).endswith("-01")]]
    expr = expr[~expr.index.duplicated(keep="first")].astype(np.float32)
    top = expr.var(axis=1).sort_values(ascending=False, kind="stable").index[:cfg.n_features]
    X = expr.loc[top].T
    sub = pd.read_csv(cfg.label_file, sep="\t", index_col=0)
    cancer = sub["Subtype_Selected"].astype(str).str.split(".").str[0]
    lab = sub.loc[cancer == cfg.subtype_cancer, cfg.label_col or "Subtype_mRNA"].reindex(X.index)
    lab = lab.astype(str).str.strip()
    lab = lab.where(~lab.isin(["nan", "", "NA", "unclassified", "-"]), other="")
    classes = sorted(v for v in lab.unique() if v != "")
    y = np.array([classes.index(v) if v != "" else -1 for v in lab], dtype=np.int64)
    Xv = X.to_numpy(dtype=np.float32)
    if np.isnan(Xv).any():
        med = np.nanmedian(Xv, axis=0)
        Xv = np.where(np.isnan(Xv), med[None, :], Xv)
    return {"X": Xv, "y": y, "feature_names": list(X.columns), "class_names": classes,
            "sample_ids": list(X.index), "n_unlabeled": int((y < 0).sum())}


def load_wdbc(cfg: DataConfig) -> dict:
    from sklearn.datasets import load_breast_cancer

    d = load_breast_cancer()
    return {"X": d.data.astype(np.float32), "y": d.target.astype(np.int64),
            "feature_names": list(d.feature_names), "class_names": list(d.target_names)}


def load_uci_tabular(cfg: DataConfig) -> dict:
    """Non-omics UCI classification data: train and test files are pooled, and the
    n_features columns of largest variance over all rows are kept (label-free), as for the cohorts."""
    raw = Path(cfg.raw_dir)
    if cfg.source == "har":
        with zipfile.ZipFile(raw / "har.zip") as outer:
            inner = zipfile.ZipFile(io.BytesIO(outer.read("UCI HAR Dataset.zip")))
        read = lambda n: inner.read(f"UCI HAR Dataset/{n}").decode()
        X = np.vstack([np.loadtxt(io.StringIO(read(f"{s}/X_{s}.txt"))) for s in ("train", "test")])
        y = np.concatenate([np.loadtxt(io.StringIO(read(f"{s}/y_{s}.txt")), dtype=int) for s in ("train", "test")])
        names = [line.split(" ", 1)[1] for line in read("features.txt").splitlines() if line.strip()]
        classes = [line.split(" ", 1)[1] for line in read("activity_labels.txt").splitlines() if line.strip()]
        y = y - 1
    elif cfg.source == "optdigits":
        with zipfile.ZipFile(raw / "optdigits.zip") as z:
            a = np.vstack([np.loadtxt(io.StringIO(z.read(n).decode()), delimiter=",")
                           for n in ("optdigits.tra", "optdigits.tes")])
        X, y = a[:, :-1], a[:, -1].astype(int)
        names, classes = [f"block_{i}" for i in range(X.shape[1])], [str(c) for c in range(10)]
    elif cfg.source == "satimage":
        with zipfile.ZipFile(raw / "satimage.zip") as z:
            a = np.vstack([np.loadtxt(io.StringIO(z.read(n).decode())) for n in ("sat.trn", "sat.tst")])
        X, codes = a[:, :-1], a[:, -1].astype(int)
        levels = sorted(set(codes.tolist()))
        y = np.array([levels.index(c) for c in codes])
        names, classes = [f"band{i % 4 + 1}_px{i // 4 + 1}" for i in range(X.shape[1])], [str(c) for c in levels]
    elif cfg.source == "mfeat":
        sets = ("fou", "fac", "kar", "pix", "zer", "mor")
        with zipfile.ZipFile(raw / "mfeat.zip") as z:
            listed = z.namelist()
            blocks = [np.loadtxt(io.StringIO(z.read([n for n in listed if n.endswith(f"mfeat-{k}")][0]).decode()))
                      for k in sets]
        X = np.hstack(blocks)
        y = np.repeat(np.arange(10), 200)
        names = [f"{k}_{i}" for k, b in zip(sets, blocks) for i in range(b.shape[1])]
        classes = [str(c) for c in range(10)]
    else:
        raise ValueError(f"unknown uci_tabular source {cfg.source}")
    if X.shape[1] > cfg.n_features:
        keep = np.sort(np.argsort(-X.var(axis=0), kind="stable")[: cfg.n_features])
        X, names = X[:, keep], [names[i] for i in keep]
    return {"X": X.astype(np.float32), "y": y.astype(np.int64), "feature_names": names,
            "class_names": classes, "sample_ids": [f"{cfg.source}_{i}" for i in range(len(y))]}


def build_dataset(cfg: DataConfig, data_root: str | Path, force: bool = False) -> dict:
    """Build or load the cached processed dataset."""
    out_dir = Path(data_root) / cfg.name
    npz = out_dir / "data.npz"
    meta_path = out_dir / "meta.json"
    if npz.exists() and meta_path.exists() and not force:
        return load_cached(out_dir)
    if cfg.kind == "metabric":
        d = load_metabric(cfg)
    elif cfg.kind == "metabric_semisynth":
        base = build_dataset(DataConfig(name="metabric", kind="metabric", n_features=cfg.n_features,
                                        raw_dir=cfg.raw_dir), data_root, force=False)
        d = dict(base)
        sim = make_semisynth(base["X"], cfg.n_causal, cfg.n_classes, cfg.target_bayes_acc,
                             n_clusters=cfg.n_groups, seed=cfg.sim_seed)
        d.update(sim)
    elif cfg.kind == "synth_scm":
        d = make_synth_scm(cfg.n_samples, cfg.n_groups, cfg.group_size, cfg.n_causal,
                           cfg.n_classes, cfg.target_bayes_acc, tuple(cfg.within_corr), cfg.sim_seed)
    elif cfg.kind == "uci_rnaseq":
        d = load_uci_rnaseq(cfg)
    elif cfg.kind == "wdbc":
        d = load_wdbc(cfg)
    elif cfg.kind == "tcga_brca":
        d = load_tcga_brca(cfg)
    elif cfg.kind == "tcga_subtype":
        d = load_tcga_subtype(cfg)
    elif cfg.kind == "uci_tabular":
        d = load_uci_tabular(cfg)
    else:
        raise ValueError(f"unknown dataset kind {cfg.kind}")
    if len(d["X"]) > cfg.max_samples:
        rng = np.random.default_rng(cfg.sim_seed)
        idx = np.sort(rng.choice(len(d["X"]), size=cfg.max_samples, replace=False))
        for k in ("X", "y", "cohort", "sample_ids"):
            if k in d:
                d[k] = [d[k][i] for i in idx] if isinstance(d[k], list) else d[k][idx]
    save_cached(d, out_dir)
    return load_cached(out_dir)


ARRAY_KEYS = ("X", "y", "group", "causal", "cohort")


def save_cached(d: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(d[k]) for k in ARRAY_KEYS if k in d}
    np.savez_compressed(out_dir / "data.npz", **arrays)
    meta = {k: v for k, v in d.items() if k not in ARRAY_KEYS}
    meta["shape"] = list(d["X"].shape)
    meta["class_counts"] = np.bincount(d["y"][d["y"] >= 0]).tolist()
    with open(out_dir / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=1, default=str)


def load_cached(out_dir: Path) -> dict:
    arrays = np.load(out_dir / "data.npz")
    d = {k: arrays[k] for k in arrays.files}
    with open(out_dir / "meta.json") as fh:
        d.update(json.load(fh))
    return d
