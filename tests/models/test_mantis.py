import numpy as np
import pytest
import torch

from tor.models import get_adapter
from tor.train import count_params

pytestmark = pytest.mark.slow

NAMES = ["mantis8m", "mantisv2"]


@pytest.mark.parametrize("name", NAMES)
def test_mantis_params_scratch_lora(name):
    a = get_adapter(name)()
    a.load("pretrained")
    total, _ = count_params(a._backbone)
    print(f"[{name}] pretrained backbone params: {total}")
    assert 3_000_000 < total < 10_000_000

    pre = {k: v.clone() for k, v in a._backbone.state_dict().items()}
    a.load(("random", 0))
    rnd = a._backbone.state_dict()
    key = next(k for k in pre if k.endswith("transf_unit.transformer.layers.0.0.fn.to_qkv.weight")
               or k.endswith("attention.wQKV.weight"))
    assert not torch.equal(pre[key], rnd[key])
    # same seed -> same random init
    a.load(("random", 0))
    assert torch.equal(rnd[key], a._backbone.state_dict()[key])
    a.load(("random", 1))
    assert not torch.equal(rnd[key], a._backbone.state_dict()[key])

    a.load("pretrained")
    counts = {}
    for mode in ("head", "lora", "full"):
        counts[mode] = count_params(a.classifier(3, mode, 1, 512))[1]
    head_model = a.classifier(3, "head", 1, 512)
    assert count_params(head_model.backbone)[1] == 0
    a._backbone.train()
    a.embed(a.preprocess(np.zeros((2, 1, 64))))
    assert not a._backbone.training
    print(f"[{name}] trainable params (3 classes, 1 channel): {counts}")
    assert counts["head"] < counts["lora"] < counts["full"]


@pytest.mark.parametrize("name", NAMES)
def test_mantis_preprocess_resizes_to_512(name):
    a = get_adapter(name)()
    for L in (96, 512, 700):
        x = a.preprocess(np.random.default_rng(0).standard_normal((2, 3, L)))
        assert tuple(x.shape) == (2, 3, 512) and x.dtype == torch.float32
    assert tuple(a.preprocess(np.zeros((2, 100))).shape) == (2, 1, 512)


@pytest.mark.parametrize("name", NAMES)
def test_mantis_embed_per_channel(name):
    a = get_adapter(name)()
    a.load("pretrained")
    x = a.preprocess(np.random.default_rng(0).standard_normal((2, 3, 64)))
    assert tuple(a.embed(x).shape) == (2, 3 * 256)
