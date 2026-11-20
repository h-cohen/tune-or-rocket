import json
import os

import numpy as np
import pytest
import torch

from tor.models import get_adapter
from tor.train import count_params

pytestmark = pytest.mark.slow

KEY = "model.transformer.layers.0.attention.wQKV.weight"


def _adapter():
    a = get_adapter("toto")()
    a.load("pretrained")
    return a


def test_toto_params_weights_scratch_lora():
    from safetensors import safe_open

    from toto.model.toto import Toto

    from tor.models.toto import _snapshot_dir

    a = _adapter()
    total, _ = count_params(a._model)
    unused = a.unused_param_count()
    print(f"[toto] pretrained params (with unembed + Student-T head): {total}; unused by classifier: {unused}")
    assert abs(total - 151_000_000) / 151_000_000 < 0.01  # paper: "151 million parameters"

    # every checkpoint tensor (after the package's own key remap) equals the loaded one
    sd = a._model.state_dict()
    n = 0
    with safe_open(os.path.join(_snapshot_dir(), "model.safetensors"), "pt") as f:
        remap = Toto._map_state_dict_keys({k: f.get_tensor(k) for k in f.keys()}, False)
    for k, v in remap.items():
        if k in sd and not k.endswith("rotary_emb.freqs"):
            assert torch.equal(sd[k], v.to(sd[k].dtype)), k
            n += 1
    assert n == len(sd) and KEY in remap  # no loaded tensor was left at its random init

    pre = sd[KEY].clone()
    a.load(("random", 0))
    rnd = a._model.state_dict()[KEY].clone()
    assert not torch.equal(pre, rnd)
    a.load(("random", 0))
    assert torch.equal(rnd, a._model.state_dict()[KEY])
    a.load(("random", 1))
    assert not torch.equal(rnd, a._model.state_dict()[KEY])

    a.load("pretrained")
    counts = {m: count_params(a.classifier(3, m, 2, 128))[1] for m in ("head", "lora", "full")}
    print(f"[toto] trainable params (3 classes, 2 channels): {counts}")
    assert counts["head"] == 768 * 2 * 3 + 3
    assert counts["full"] == total - unused + counts["head"]
    assert counts["head"] < counts["lora"] < counts["full"]
    ftot, ftr = count_params(a.classifier(3, "full", 1, 128))
    assert ftot == ftr
    assert count_params(a.classifier(3, "head", 1, 128).backbone)[1] == 0
    lm = a.classifier(3, "lora", 1, 128)
    names = [n for n, p in lm.backbone.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n and (".attention.wQKV." in n or ".attention.wO." in n) for n in names)
    assert len(names) == 12 * 2 * 2  # time-wise (11) and space-wise (1) layers, both projections, A and B


def test_toto_layer_types():
    a = _adapter()
    axes = [str(layer.attention_axis) for layer in a._model.model.transformer.layers]
    assert axes == ["AttentionAxis.TIME"] * 11 + ["AttentionAxis.SPACE"]


def test_toto_preprocess():
    a = get_adapter("toto")()
    x = a.preprocess(np.random.default_rng(0).standard_normal((2, 3, 100)))
    assert tuple(x.shape) == (2, 3, 100) and x.dtype == torch.float32
    L = a.max_length
    z = a.preprocess(np.arange(L + 50, dtype="float32")[None, None])
    assert tuple(z.shape) == (1, 1, L) and z[0, 0, -1] == L + 49 and z[0, 0, 0] == 50
    for v in (np.nan, np.inf):
        bad = np.zeros((2, 1, 64))
        bad[0, 0, 3] = v
        with pytest.raises(ValueError):
            a.preprocess(bad)


def test_toto_hidden_states_equal_package_backbone():
    a = _adapter()
    x = a.preprocess(np.random.default_rng(0).standard_normal((2, 3, 100)))  # 100 -> pad 28 -> 2 patches
    bb = a._model.model
    captured = []
    h = bb.transformer.register_forward_hook(lambda mod, i, o: captured.append(o))
    series, real = a._backbone.pad(x)
    with torch.no_grad():
        bb.backbone(series, real, torch.ones_like(series))  # package's own path (scaler, patch_embed, transformer)
        hidden, valid = a._backbone.tokens(x)
    h.remove()
    assert hidden.shape[2] == 2 and torch.equal(hidden, captured[0])
    assert valid.all()
    want = hidden.mean(dim=2).flatten(1)
    assert tuple(a.embed(x).shape) == (2, 3 * 768)
    assert torch.allclose(a.embed(x), want, atol=1e-6)


def test_toto_fully_padded_patches_excluded():
    a = _adapter()
    x = torch.randn(2, 2, 64)
    real = torch.ones_like(x, dtype=torch.bool)
    # emulate two extra fully padded leading patches at the package level
    ext = torch.cat([torch.zeros(2, 2, 128), x], dim=-1)
    ereal = torch.cat([torch.zeros(2, 2, 128, dtype=torch.bool), real], dim=-1)
    bb = a._model.model
    with torch.no_grad():
        cap = []
        hk = bb.transformer.register_forward_hook(lambda m, i, o: cap.append(o))
        bb.backbone(ext, ereal, torch.ones_like(ext))
        hk.remove()
        valid = ereal.unfold(-1, 64, 64).any(-1)
    assert tuple(valid.shape) == (2, 2, 3) and valid.sum() == 2 * 2 * 1
    assert not cap[0].isnan().any()
    pooled = (cap[0] * valid.unsqueeze(-1)).sum(2) / valid.sum(2, keepdim=True)
    assert torch.allclose(pooled, cap[0][:, :, -1], atol=1e-6)  # only the real patch contributes


def test_toto_native_multivariate_space_attention():
    a = _adapter()
    x = a.preprocess(np.random.default_rng(0).standard_normal((2, 2, 64)))
    e = a.embed(x)
    assert tuple(e.shape) == (2, 2 * 768) and not e.isnan().any()
    x2 = x.clone()
    x2[:, 1] = torch.randn(2, 64)
    # channels are fed as variates: changing channel 1 changes channel 0's embedding (space-wise layer)
    assert not torch.allclose(e[:, :768], a.embed(x2)[:, :768], atol=1e-5)


def test_toto_short_and_constant_inputs_finite():
    a = _adapter()
    short = a.preprocess(np.random.default_rng(0).standard_normal((2, 1, 5)))
    const = torch.full((2, 1, 64), 3.0)
    for x in (short, const):
        e = a.embed(x)
        assert tuple(e.shape) == (2, 768) and torch.isfinite(e).all()
