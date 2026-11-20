import numpy as np
import pytest
import torch

from tor.models import available_adapters, get_adapter
from tor.train import count_params

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("name", available_adapters())
def test_adapter_contract(name):
    a = get_adapter(name)()
    rng = np.random.default_rng(0)
    fracs = {}
    for mode in a.supported_modes:
        a.load(("random", 0) if mode == "scratch" else "pretrained")
        for c in (1, 3):
            X = rng.standard_normal((4, c, 128)).astype("float32")
            x = a.preprocess(X)
            assert x.dtype == torch.float32 and x.ndim == 3
            model = a.classifier(3, mode, c, x.shape[-1])
            logits = model.eval()(x)
            assert tuple(logits.shape) == (4, 3)
        tot, tr = count_params(model)
        fracs[mode] = tr / tot
        if mode in ("full", "scratch"):
            assert fracs[mode] == 1.0
        if a.supports_grad_ckpt:
            m = a.classifier(3, mode, c, x.shape[-1], grad_ckpt=True).train()
            m(x).sum().backward()
        else:
            with pytest.raises(ValueError):
                a.classifier(3, mode, c, x.shape[-1], grad_ckpt=True)
    if {"head", "lora", "full"} <= set(fracs):
        assert fracs["head"] < fracs["lora"] < fracs["full"]
    elif {"head", "full"} <= set(fracs):
        assert fracs["head"] < fracs["full"]
    with pytest.raises(ValueError):
        a.classifier(3, "bogus", 1, 128)
    a.load(("random", 0) if a.hf_repo is None else "pretrained")
    for c in (1, 3):
        e = a.embed(a.preprocess(rng.standard_normal((4, c, 128)).astype("float32")))
        assert e.ndim == 2 and e.shape[0] == 4
