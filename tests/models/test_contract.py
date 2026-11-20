import numpy as np
import pytest
import torch

from tor.models import REGISTRY, available_adapters
from tor.train import count_params

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("name", available_adapters())
def test_adapter_contract(name):
    a = REGISTRY[name]()
    X = np.random.default_rng(0).standard_normal((4, 1, 128)).astype("float32")
    fracs = {}
    for mode in a.supported_modes:
        if mode != "scratch":
            a.load("pretrained")
        else:
            a.load(("random", 0))
        x = a.preprocess(X)
        model = a.classifier(3, mode)
        logits = model.eval()(x)
        assert tuple(logits.shape) == (4, 3)
        tr, tot = count_params(model)
        fracs[mode] = tr / tot
    if {"head", "lora", "full"} <= set(fracs):
        assert fracs["head"] < fracs["lora"] < fracs["full"]
    elif {"head", "full"} <= set(fracs):
        assert fracs["head"] < fracs["full"]
    if a.hf_repo is not None:  # embed needs pretrained weights
        a.load("pretrained")
        assert a.embed(a.preprocess(X)).ndim == 2
    else:  # no pretrained weights: embed is GAP features of the current net
        a.load(("random", 0))
        assert a.embed(a.preprocess(X)).ndim == 2
