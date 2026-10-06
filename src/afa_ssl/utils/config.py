"""YAML-backed dataclass configs. No hardcoded hyperparameters elsewhere."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    name: str
    kind: str  # metabric | metabric_semisynth | synth_scm | uci_rnaseq | wdbc
    n_features: int = 200
    n_test: int = 400
    max_samples: int = 2000
    label_col: str | None = None
    label_file: str = "patient"  # metabric: which clinical file holds label_col (patient | sample)
    raw_dir: str | None = None
    raw_url: str | None = None
    subtype_cancer: str | None = None  # tcga_subtype: cancer key in the PanCanAtlas subtype table
    source: str | None = None  # uci_tabular: har | optdigits | satimage | mfeat
    # synthetic / semi-synthetic knobs
    n_samples: int = 2000
    n_groups: int = 10
    group_size: int = 10
    n_causal: int = 5
    n_classes: int = 3
    target_bayes_acc: float = 0.85
    within_corr: tuple[float, float] = (0.5, 0.85)
    sim_seed: int = 12345
    # pool corruption (robustness ablation)
    pool_missing_rate: float = 0.0
    test_noise_sd: float = 0.0
    # cohort restriction for transfer experiment: pretrain on these cohorts
    pretrain_cohorts: list[int] | None = None
    eval_cohorts: list[int] | None = None
    # tuning mode: evaluate on validation rows carved out of the pool; the real test rows are unused
    tune_mode: bool = False
    n_val: int = 300


@dataclass
class ModelConfig:
    hidden: int = 128
    latent: int = 64
    n_layers: int = 2
    dropout: float = 0.0
    head: str = "linear"  # linear | mlp
    ssl_objective: str = "gaussian_nll"  # gaussian_nll | mse (homoscedastic ablation)
    logvar_min: float = -6.0
    logvar_max: float = 4.0


@dataclass
class MaskConfig:
    budget_mix: float = 0.5  # prob. of drawing k from the acquisition regime
    budget_max: int = 20     # acquisition regime k ~ U{0..budget_max}


@dataclass
class TrainConfig:
    ssl_steps: int = 3000
    ssl_lr: float = 1e-3
    ssl_wd: float = 1e-5
    ssl_batch: int = 128
    ssl_val_frac: float = 0.1
    ssl_eval_every: int = 100
    ssl_pool_frac: float = 1.0  # fraction of the unlabeled pool used for pretraining
    sup_steps: int = 1500
    sup_lr: float = 1e-3
    sup_wd: float = 1e-3
    sup_batch: int = 64
    head_steps: int = 1500
    head_lr: float = 3e-3
    head_wd: float = 1e-3
    head_batch: int = 64
    finetune: bool = False        # also update a copy of the pretrained encoder with the labels
    ft_lr_encoder: float = 1e-4   # encoder learning rate while fine-tuning (head keeps head_lr)
    distill_weight: float = 0.0   # partial-to-full self-distillation on unlabeled pool rows
    distill_batch: int = 128
    # mixture weight of the pretrained member in the predictive ensemble, chosen by masked
    # cross-validation on the labeled rows over `mix_grid` (mix_folds = 0 fixes it to mix_fixed)
    mix_folds: int = 3
    mix_grid: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    mix_masks: int = 20
    mix_fixed: float = 0.5
    # self-training on the unlabeled pool: rounds of pseudo-labelling confident full-view predictions
    st_rounds: int = 0
    st_threshold: float = 0.7
    st_weight: float = 0.5   # loss weight of pseudo-labeled rows relative to labeled rows
    max_minutes: float = 30.0


@dataclass
class BaselineConfig:
    """Published baselines: EDDI (partial VAE) and DIME (value network)."""
    pvae_emb: int = 16
    pvae_hidden: int = 128
    pvae_latent: int = 32
    pvae_lr: float = 1e-3
    pvae_wd: float = 1e-5
    pvae_steps: int = 3000
    pvae_batch: int = 128
    pvae_eval_every: int = 100
    pvae_latent_samples: int = 1
    pvae_label_weight: float | None = None   # None means D (see models/eddi.py)
    pvae_label_obs_prob: float = 0.5
    dime_hidden: int = 128
    dime_lr: float = 1e-3
    dime_wd: float = 1e-4
    dime_steps: int = 1500
    dime_batch: int = 64


@dataclass
class AcqConfig:
    budget: int = 20
    gh_nodes: int = 3
    cost_model: str = "uniform"  # uniform | tiers
    cost_tiers: list[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])
    cost_budget: float | None = None  # in cost units; default budget * mean cost
    chunk_rows: int = 64
    methods: list[str] = field(default_factory=lambda: [
        "rand", "unc", "eig_sup", "er_sup", "ecd_sup", "ssl_rand", "ssl_unc", "ours",
    ])


@dataclass
class RunConfig:
    experiment: str
    data: DataConfig
    seed: int = 0
    n_labels: int = 50
    model: ModelConfig = field(default_factory=ModelConfig)
    sup_model: ModelConfig = field(default_factory=ModelConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    acq: AcqConfig = field(default_factory=AcqConfig)
    threads: int = 4
    max_ram_gb: float = 6.0
    out_root: str = "experiments"
    log_root: str = "logs"
    data_root: str = "data/processed"
    tag: str = ""

    @property
    def run_id(self) -> str:
        base = f"{self.experiment}_{self.data.name}_s{self.seed}_n{self.n_labels}"
        return base + (f"_{self.tag}" if self.tag else "")


def _from_dict(cls, d: dict[str, Any]):
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        if dataclasses.is_dataclass(f.type) if isinstance(f.type, type) else False:
            v = _from_dict(f.type, v)
        elif f.name in ("data",):
            v = _from_dict(DataConfig, v)
        elif f.name in ("model", "sup_model"):
            v = _from_dict(ModelConfig, v)
        elif f.name == "mask":
            v = _from_dict(MaskConfig, v)
        elif f.name == "train":
            v = _from_dict(TrainConfig, v)
        elif f.name == "acq":
            v = _from_dict(AcqConfig, v)
        elif f.name == "baseline":
            v = _from_dict(BaselineConfig, v)
        kwargs[f.name] = v
    return cls(**kwargs)


def deep_update(base: dict, upd: dict) -> dict:
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str | Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def to_dict(cfg) -> dict:
    return dataclasses.asdict(cfg)


def config_hash(cfg) -> str:
    s = json.dumps(to_dict(cfg), sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def build_run_config(
    experiment_yaml: str | Path,
    dataset_yaml: str | Path,
    overrides: dict | None = None,
    defaults_yaml: str | Path = "configs/default.yaml",
) -> RunConfig:
    """defaults < experiment < overrides; the dataset block comes from its own file."""
    d = load_yaml(defaults_yaml)
    d = deep_update(d, load_yaml(experiment_yaml))
    d = deep_update(d, overrides or {})
    d["data"] = deep_update(load_yaml(dataset_yaml), d.get("data", {}) or {})
    return _from_dict(RunConfig, d)


def save_config(cfg, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(to_dict(cfg), fh, sort_keys=False)
