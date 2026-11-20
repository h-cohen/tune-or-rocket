"""Fake adapters for probe tests (importable in spawned children via TOR_PROBE_EXTRA_MODULES)."""
import os
import time

import torch

from tor.models import register
from tor.models.base import FMAdapter, Recipe


EVENTS = []


class _Net(torch.nn.Module):
    def __init__(self, n_classes, c, oom_batch=None, oom_infer=False, sleep=0.0):
        super().__init__()
        self.conv = torch.nn.Conv1d(c, 4, 3)
        self.head = torch.nn.Linear(4, n_classes)
        self.oom_batch, self.oom_infer, self.sleep = oom_batch, oom_infer, sleep

    def forward(self, x):
        if self.sleep:
            time.sleep(self.sleep)
        if (self.oom_batch is not None and x.shape[0] >= self.oom_batch) or (
                self.oom_infer and not self.training):
            EVENTS.append("oom")
            raise torch.cuda.OutOfMemoryError("fake oom")
        return self.head(self.conv(x).mean(-1))


class FakeAdapter(FMAdapter):
    name = "fake_probe"
    hf_repo = None
    max_length = None
    multivariate = "per_channel"
    pretraining_corpora = "none"
    recipe_source = "test"
    supported_modes = ("head", "full")
    supports_grad_ckpt = True
    oom_batch = None  # OOM in forward when batch >= this
    oom_without_ckpt = False
    oom_infer = False
    sleep = 0.0

    def load(self, init):
        pass

    def preprocess(self, X):
        x = torch.as_tensor(X, dtype=torch.float32)
        return x[..., :32]  # truncates: preprocessed length differs from raw

    def embed(self, X):
        return X.mean(-1)

    def classifier(self, n_classes, mode, n_channels, seq_len, grad_ckpt=False, seed=0):
        self._check_mode(mode, grad_ckpt)
        assert seq_len <= 32
        if self.oom_without_ckpt and not grad_ckpt:
            raise torch.cuda.OutOfMemoryError("fake oom (no ckpt)")
        return _Net(n_classes, n_channels, self.oom_batch, self.oom_infer, self.sleep)

    def default_recipe(self):
        return Recipe(lr=1e-3, weight_decay=0.0, epochs=3, batch_size=8)

    def unused_param_count(self):
        return 7


class NoCkptAdapter(FakeAdapter):
    name = "fake_nockpt"
    supports_grad_ckpt = False
    oom_batch = 1


class SleepAdapter(FakeAdapter):
    name = "fake_sleep"

    def classifier(self, *a, **k):
        time.sleep(120)


class SlowAdapter(FakeAdapter):
    name = "fake_slow"
    sleep = 1.0


register(FakeAdapter)
register(SleepAdapter)
register(SlowAdapter)
if os.environ.get("TOR_PROBE_FAKES_ONLY"):  # skip importing real adapters in children
    import tor.models as _m

    _m._import_all = lambda: None
