"""InceptionTime reference (from scratch only), in PyTorch.

6 inception modules (3 per residual block), 32 filters, kernels 40/20/10, GAP head.
Only the `scratch` mode exists. `embed` returns GAP features (eval mode) of a
randomly initialised network built lazily from the seed given to `load`.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from tor.models import register
from tor.models.base import FMAdapter, Mode, Recipe

KERNELS = (40, 20, 10)
FILTERS = 32


class InceptionModule(nn.Module):
    def __init__(self, c_in: int):
        super().__init__()
        self.bottleneck = nn.Conv1d(c_in, FILTERS, 1, bias=False) if c_in > 1 else nn.Identity()
        b = FILTERS if c_in > 1 else c_in
        self.convs = nn.ModuleList(
            nn.Conv1d(b, FILTERS, k, padding="same", bias=False) for k in KERNELS
        )
        self.pool_conv = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(c_in, FILTERS, 1, bias=False)
        )
        self.bn = nn.BatchNorm1d(FILTERS * 4)
        self.act = nn.ReLU()

    def forward(self, x):
        z = self.bottleneck(x)
        out = torch.cat([c(z) for c in self.convs] + [self.pool_conv(x)], dim=1)
        return self.act(self.bn(out))


class InceptionNet(nn.Module):
    def __init__(self, c_in: int, n_classes: int | None = None, depth: int = 6):
        super().__init__()
        d = FILTERS * 4
        self.modules_ = nn.ModuleList(
            InceptionModule(c_in if i == 0 else d) for i in range(depth)
        )
        # residual every 3 modules; shortcut from the block input
        self.shortcuts = nn.ModuleDict()
        for i in range(2, depth, 3):
            src = c_in if i == 2 else d
            self.shortcuts[str(i)] = nn.Sequential(
                nn.Conv1d(src, d, 1, bias=False), nn.BatchNorm1d(d)
            )
        self.feature_dim = d
        self.in_channels = c_in
        self.head = nn.Linear(d, n_classes) if n_classes else None

    def features(self, x):
        res = x
        for i, m in enumerate(self.modules_):
            x = m(x)
            if str(i) in self.shortcuts:
                x = torch.relu(x + self.shortcuts[str(i)](res))
                res = x
        return x.mean(dim=-1)  # GAP

    def forward(self, x):
        return self.head(self.features(x))


@register
class InceptionTime(FMAdapter):
    name = "inceptiontime"
    hf_repo = None
    max_length = None
    multivariate = "native"
    pretraining_corpora = "none (trained from scratch)"
    recipe_source = "Ismail Fawaz et al. 2020, InceptionTime (Data Min Knowl Disc 34)"
    supported_modes: tuple[Mode, ...] = ("scratch",)

    def __init__(self):
        self._seed = 0
        self._net: InceptionNet | None = None  # lazily built for embed

    def load(self, init) -> None:
        if init == "pretrained":
            raise ValueError("inceptiontime has no pretrained weights")
        kind, seed = init
        if kind != "random":
            raise ValueError(f"bad init {init!r}")
        self._seed = seed
        self._net = None  # built at first embed() once the channel count is known

    def preprocess(self, X: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        if x.ndim == 2:
            x = x[:, None, :]
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp_min(1e-8)
        return (x - mu) / sd

    def embed(self, X: torch.Tensor) -> torch.Tensor:
        if self._net is None or self._net.in_channels != X.shape[1]:
            self._net = self._build(X.shape[1], None, self._seed)
        self._net.eval()
        with torch.no_grad():
            return self._net.features(X)

    @staticmethod
    def _build(n_channels, n_classes, seed):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return InceptionNet(n_channels, n_classes)

    def classifier(
        self, n_classes, mode, n_channels, seq_len, grad_ckpt=False, seed=0
    ) -> torch.nn.Module:
        self._check_mode(mode, grad_ckpt)
        return self._build(n_channels, n_classes, seed)

    def default_recipe(self) -> Recipe:
        return Recipe(lr=1e-3, weight_decay=0.0, epochs=1500, batch_size=64)
