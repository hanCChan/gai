"""Encoder wrappers with a fixed feature layer (EXPERIMENT_PROTOCOL.md §2.3).

    wrapper.encode(x)        -> z (B, feature_dim)       the representation every head uses
    wrapper.native_logits(x) -> (B, num_classes)          identical to the original model's forward
    wrapper.linear_head()    -> (W, b) of the native classifier acting on z

WiSig OfficialCNN: z = encode() = 80-D post-ReLU fc2 (before dropout); native = head(z) in eval mode.
Shen FeatureCNN:   z = 128-D relu(tx_input(L2-normalised 512-D feature)) = the classifier INPUT,
                   captured with a forward pre-hook so the original forward code is executed unchanged.
                   The 512-D normalised feature is available only as `historical_512d` (diagnostic).
New identities (OSU 25 classes): the 10-class classifier is NOT reused. `new_head=True` attaches a fresh,
explicitly sized Linear on the frozen trunk; without it a class-count mismatch raises.
"""
from __future__ import annotations

import sys

import torch
from torch import nn

from .registry import WORKTREE


def _scripts_on_path():
    for p in (WORKTREE / 'scripts', WORKTREE / 'src'):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


class EncoderWrapper(nn.Module):
    feature_dim: int
    num_classes: int
    feature_layer: str

    def linear_head(self):
        lin = self._head()
        return lin.weight.detach().cpu().double(), lin.bias.detach().cpu().double()


class WiSigOfficialEncoder(EncoderWrapper):
    feature_dim = 80
    feature_layer = 'OfficialCNN.encode (fc2 post-ReLU, pre-dropout), 80-D'

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_classes = model.head.out_features

    def _check(self, x):
        if tuple(x.shape[1:]) != (1, 256, 2):
            raise ValueError(f'WiSig canonical input is float32[B,1,256,2], got {tuple(x.shape)}')

    def encode(self, x):
        self._check(x)
        return self.model.encode(x)

    def native_logits(self, x):
        return self.model.head(self.encode(x))

    def train_forward(self, x):
        """One pass for training: (z, native logits with the original dropout in train mode)."""
        self._check(x)
        z = self.model.encode(x)
        return z, self.model.head(self.model.drop(z))

    def regularization(self):
        return self.model.regularization()

    def _head(self):
        return self.model.head


class ShenCISEncoder(EncoderWrapper):
    feature_dim = 128
    feature_layer = 'FeatureCNN classifier input = relu(tx_input(normalize(feature))), 128-D'

    def __init__(self, model, num_classes=None, new_head=False):
        super().__init__()
        self.model = model
        native = model.classifier.out_features
        num_classes = native if num_classes is None else int(num_classes)
        if num_classes != native:
            if not new_head:
                raise ValueError(f'checkpoint classifier has {native} classes, task has {num_classes}; '
                                 f'pass new_head=True to attach a fresh head on the frozen trunk')
            self.model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        self.num_classes = num_classes
        self._cache = {}
        self.model.classifier.register_forward_pre_hook(self._grab)

    def _grab(self, module, inputs):
        self._cache['z'] = inputs[0]

    def _run(self, x):
        logits, feat512 = self.model.forward_with_features(x)
        return logits, self._cache.pop('z'), feat512

    def encode(self, x):
        return self._run(x)[1]

    def native_logits(self, x):
        return self._run(x)[0]

    def historical_512d(self, x):
        return self._run(x)[2]

    def train_forward(self, x):
        logits, z, _ = self._run(x)
        return z, logits

    def regularization(self):
        return 0.

    def _head(self):
        return self.model.classifier


def build_wisig_official(state_dict=None, classes=10):
    _scripts_on_path()
    from wisig_w01_common import OfficialCNN
    m = OfficialCNN(classes)
    if state_dict is not None:
        m.load_state_dict(state_dict)
    return WiSigOfficialEncoder(m.eval())


def build_shen_featurecnn(state_dict=None, num_classes=None, new_head=False):
    _scripts_on_path()
    from b1_baselines.model import FeatureCNN
    m = FeatureCNN()
    if state_dict is not None:
        m.load_state_dict(state_dict)
    return ShenCISEncoder(m.eval(), num_classes=num_classes, new_head=new_head)


@torch.no_grad()
def extract(wrapper: EncoderWrapper, x, batch=1024, device='cpu'):
    """Eval-mode feature extraction (BatchNorm/Dropout frozen). Returns float64 numpy (N, d)."""
    import numpy as np
    wrapper = wrapper.to(device).eval()
    out = []
    for i in range(0, len(x), batch):
        xb = torch.as_tensor(np.ascontiguousarray(x[i:i + batch]), dtype=torch.float32, device=device)
        out.append(wrapper.encode(xb).double().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, wrapper.feature_dim))
