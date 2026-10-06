import numpy as np
import torch

from afa_ssl.models.mfa import MFA
from afa_ssl.train.loops import train_head
from afa_ssl.utils.config import MaskConfig, ModelConfig, TrainConfig


def test_finetune_leaves_mfa_untouched():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    X = rng.normal(size=(40, 6)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    mfa = MFA(6, 8, 4)
    before = {k: v.clone() for k, v in mfa.state_dict().items()}
    tcfg = TrainConfig(head_steps=20, finetune=True, ft_lr_encoder=1e-2)
    clf, _ = train_head(mfa, X, y, 2, ModelConfig(), tcfg, MaskConfig(), rng)
    for k, v in mfa.state_dict().items():
        assert torch.equal(v, before[k]), k
    assert clf.encoder is not mfa.encoder
    changed = any(not torch.equal(a, b) for a, b in zip(clf.encoder.state_dict().values(),
                                                        mfa.encoder.state_dict().values()))
    assert changed
