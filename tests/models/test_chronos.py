import numpy as np
import pytest
import torch

from tor.models import get_adapter
from tor.train import count_params

pytestmark = pytest.mark.slow

# (name, d_model, expected total params of the loaded pretrained model, key checked vs checkpoint)
CASES = [
    ("chronos-bolt-small", 512, 47_718_016, "encoder.block.0.layer.0.SelfAttention.q.weight"),
    ("chronos-bolt-base", 768, 205_292_928, "encoder.block.0.layer.0.SelfAttention.q.weight"),
    ("chronos2", 768, 119_477_664, "encoder.block.0.layer.0.self_attention.q.weight"),
]


@pytest.mark.parametrize("name,d,total_exp,key", CASES)
def test_chronos_params_weights_scratch_lora(name, d, total_exp, key):
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    a = get_adapter(name)()
    a.load("pretrained")
    total, _ = count_params(a._model)
    unused = a.unused_param_count()
    print(f"[{name}] pretrained params (with decoder/forecast head): {total}; unused by classifier: {unused}")
    assert total == total_exp

    from tor.models.chronos import _hub_kwargs

    path = hf_hub_download(a.hf_repo, "model.safetensors", **_hub_kwargs())
    with safe_open(path, "pt") as f:
        ck = f.get_tensor(key)
    pre = a._model.state_dict()[key].clone()
    assert torch.equal(pre, ck.to(pre.dtype))  # weights really loaded from the checkpoint

    a.load(("random", 0))
    rnd = a._model.state_dict()[key].clone()
    assert not torch.equal(pre, rnd)
    a.load(("random", 0))
    assert torch.equal(rnd, a._model.state_dict()[key])
    a.load(("random", 1))
    assert not torch.equal(rnd, a._model.state_dict()[key])

    a.load("pretrained")
    counts = {m: count_params(a.classifier(3, m, 1, 128))[1] for m in ("head", "lora", "full")}
    print(f"[{name}] trainable params (3 classes, 1 channel): {counts}")
    assert counts["head"] == d * 3 + 3
    assert counts["full"] == total - unused + counts["head"]
    assert counts["head"] < counts["lora"] < counts["full"]
    ftot, ftr = count_params(a.classifier(3, "full", 1, 128))
    assert ftot == ftr
    assert count_params(a.classifier(3, "head", 1, 128).backbone)[1] == 0


@pytest.mark.parametrize("name", ["chronos-bolt-small", "chronos2"])
def test_chronos_preprocess(name):
    a = get_adapter(name)()
    rng = np.random.default_rng(0)
    x = a.preprocess(rng.standard_normal((2, 3, 100)))
    assert tuple(x.shape) == (2, 3, 100) and x.dtype == torch.float32
    L = a.max_length
    z = a.preprocess(np.arange(L + 50, dtype="float32")[None, None])
    assert tuple(z.shape) == (1, 1, L) and z[0, 0, -1] == L + 49 and z[0, 0, 0] == 50  # last L kept
    bad = np.zeros((2, 1, 64))
    bad[0, 0, 3] = np.nan
    with pytest.raises(ValueError):
        a.preprocess(bad)


def test_chronos2_group_attention_mixes_channels():
    a = get_adapter("chronos2")()
    a.load("pretrained")
    rng = np.random.default_rng(0)
    x = a.preprocess(rng.standard_normal((2, 2, 64)).astype("float32"))
    e = a.embed(x)
    assert tuple(e.shape) == (2, 2 * 768)
    x2 = x.clone()
    x2[:, 1] = torch.randn(2, 64)
    e2 = a.embed(x2)
    # channel 0 embedding changes when channel 1 changes (group attention); series do not mix
    assert not torch.allclose(e[:, :768], e2[:, :768], atol=1e-4)
    x3 = x.clone()
    x3[1] = torch.randn(2, 64)
    assert torch.allclose(e[0], a.embed(x3)[0], atol=1e-4)


def test_bolt_channels_independent():
    a = get_adapter("chronos-bolt-small")()
    a.load("pretrained")
    x = a.preprocess(np.random.default_rng(0).standard_normal((2, 2, 64)))
    e = a.embed(x)
    assert tuple(e.shape) == (2, 2 * 512)
    x2 = x.clone()
    x2[:, 1] = torch.randn(2, 64)
    assert torch.allclose(e[:, :512], a.embed(x2)[:, :512], atol=1e-4)
