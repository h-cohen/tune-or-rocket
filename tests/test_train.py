import numpy as np
import pytest
import torch

import tor.train
from tor.models.base import FMAdapter, Recipe
from tor.train import count_params, train_epochs


class _ToyNet(torch.nn.Module):
    def __init__(self, n_classes, mode):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(16, 8), torch.nn.BatchNorm1d(8), torch.nn.Dropout(0.5)
        )
        self.head = torch.nn.Linear(8, n_classes)
        if mode == "head":
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        return self.head(self.encoder(x))


class ToyAdapter(FMAdapter):
    name = "toy"
    hf_repo = None
    max_length = 16
    multivariate = "per_channel"
    pretraining_corpora = "none"
    recipe_source = "test"
    supported_modes = ("head", "full")

    def load(self, init):
        pass

    def preprocess(self, X):
        return torch.as_tensor(X, dtype=torch.float32).reshape(len(X), -1)

    def embed(self, X):
        return X[:, :8]

    def classifier(self, n_classes, mode, n_channels=1, seq_len=16, grad_ckpt=False, seed=0):
        self._check_mode(mode, grad_ckpt)
        return _ToyNet(n_classes, mode)

    def default_recipe(self):
        return Recipe(lr=1e-3, weight_decay=0.0, epochs=3, batch_size=8)


def _data(n=32):
    g = torch.Generator().manual_seed(0)
    return torch.randn(n, 16, generator=g), torch.randint(0, 3, (n,), generator=g)


def test_modes_set_trainable_params():
    a = ToyAdapter()
    tot, th = count_params(a.classifier(3, "head", 1, 16))
    tot2, tf = count_params(a.classifier(3, "full", 1, 16))
    assert tot == tot2
    assert tf == tot2
    assert th == 8 * 3 + 3 < tf
    with pytest.raises(ValueError):
        a.classifier(3, "lora", 1, 16)


def test_train_epochs_records_times_and_losses():
    X, y = _data()
    m = ToyAdapter().classifier(3, "full", 1, 16)
    r = train_epochs(m, X, y, Recipe(1e-3, 0.0, 3, 8), 3, "cpu")
    assert len(r.losses) == 3 and len(r.epoch_times_s) == 3
    assert r.diverged is False


def test_divergence_detected():
    X, y = _data()
    m = ToyAdapter().classifier(3, "full", 1, 16)
    r = train_epochs(m, X * 1e35, y, Recipe(1e6, 0.0, 50, 8), 50, "cpu")
    assert r.diverged is True
    assert len(r.losses) < 50
    assert np.isnan(r.losses[-1])


def test_no_eval_api():
    assert not hasattr(tor.train, "evaluate")
    assert not hasattr(tor.train, "accuracy")


def test_grad_ckpt_unsupported_raises():
    with pytest.raises(ValueError):
        ToyAdapter().classifier(3, "full", 1, 16, grad_ckpt=True)


def test_head_mode_keeps_frozen_encoder_fixed():
    X, y = _data()
    m = ToyAdapter().classifier(3, "head", 1, 16)
    bn = m.encoder[1]
    stats = bn.running_mean.clone(), bn.running_var.clone()
    m.eval()
    ref = m.encoder(X[:4]).clone()
    train_epochs(m, X, y, Recipe(1e-2, 0.0, 1, 8), 1, "cpu")
    assert torch.equal(bn.running_mean, stats[0]) and torch.equal(bn.running_var, stats[1])
    m.eval()
    assert torch.equal(m.encoder(X[:4]), ref)


def test_registry_duplicate_and_unavailable(monkeypatch):
    import tor.models as M

    class Dup(ToyAdapter):
        name = "inceptiontime"

    M.available_adapters()
    with pytest.raises(ValueError):
        M.register(Dup)

    class Gone(ToyAdapter):
        name = "gone"

        @classmethod
        def is_available(cls):
            return False, "no dep"

    monkeypatch.setitem(M.REGISTRY, "gone", Gone)
    assert "gone" not in M.available_adapters()
