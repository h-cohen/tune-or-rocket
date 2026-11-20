import numpy as np
import pytest
import torch

from tor.models import get_adapter
from tor.train import count_params

pytestmark = pytest.mark.slow

KEY = "stacked_xf.0.attn.qkv_proj.weight"


def _adapter():
    a = get_adapter("timesfm25")()
    a.load("pretrained")
    return a


def test_timesfm_params_weights_scratch_lora():
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    from tor.models.timesfm import _REPO, _WEIGHTS, _hub_kwargs

    a = _adapter()
    total, _ = count_params(a._model)
    unused = a.unused_param_count()
    print(f"[timesfm25] pretrained params (with forecast heads): {total}; unused by classifier: {unused}")
    assert 190_000_000 < total < 250_000_000  # model card: 200M

    with safe_open(hf_hub_download(_REPO, _WEIGHTS, **_hub_kwargs()), "pt") as f:
        ck = f.get_tensor(KEY)
    pre = a._model.state_dict()[KEY].clone()
    assert torch.equal(pre, ck.to(pre.dtype))  # weights really loaded from the checkpoint

    a.load(("random", 0))
    rnd = a._model.state_dict()[KEY].clone()
    assert not torch.equal(pre, rnd)
    a.load(("random", 0))
    assert torch.equal(rnd, a._model.state_dict()[KEY])
    a.load(("random", 1))
    assert not torch.equal(rnd, a._model.state_dict()[KEY])

    a.load("pretrained")
    counts = {m: count_params(a.classifier(3, m, 1, 128))[1] for m in ("head", "lora", "full")}
    print(f"[timesfm25] trainable params (3 classes, 1 channel): {counts}")
    assert counts["head"] == 1280 * 3 + 3
    assert counts["full"] == total - unused + counts["head"]
    assert counts["head"] < counts["lora"] < counts["full"]
    ftot, ftr = count_params(a.classifier(3, "full", 1, 128))
    assert ftot == ftr
    assert count_params(a.classifier(3, "head", 1, 128).backbone)[1] == 0
    # lora hits exactly the attention projections
    lm = a.classifier(3, "lora", 1, 128)
    names = [n for n, p in lm.backbone.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n and (".attn.qkv_proj." in n or ".attn.out." in n) for n in names)
    assert len(names) == 20 * 2 * 2


def test_timesfm_preprocess():
    a = get_adapter("timesfm25")()
    x = a.preprocess(np.random.default_rng(0).standard_normal((2, 3, 100)))
    assert tuple(x.shape) == (2, 3, 100) and x.dtype == torch.float32
    L = a.max_length
    z = a.preprocess(np.arange(L + 50, dtype="float32")[None, None])
    assert tuple(z.shape) == (1, 1, L) and z[0, 0, -1] == L + 49 and z[0, 0, 0] == 50
    bad = np.zeros((2, 1, 64))
    bad[0, 0, 3] = np.nan
    with pytest.raises(ValueError):
        a.preprocess(bad)


def test_timesfm_hidden_states_equal_package_prefill_and_pooling():
    a = _adapter()
    x = a.preprocess(np.random.default_rng(0).standard_normal((2, 2, 100)))  # 100 -> pad 28 -> 4 patches
    n, c, _ = x.shape
    rows = x.reshape(n * c, -1)
    m = a._model
    captured = []
    h = m.stacked_xf[-1].register_forward_hook(lambda mod, i, o: captured.append(o[0]))
    padded, mask = a._backbone.pad(rows)
    with torch.no_grad():
        m.decode(1, padded, mask)  # package's own prefill path (horizon 1: no AR steps)
        hidden, valid = a._backbone.tokens(rows)
    h.remove()
    assert hidden.shape[1] == 4 and torch.equal(hidden, captured[0])  # exact match with package
    assert valid.all()  # the partially padded first patch is a real patch
    with torch.no_grad():
        want = hidden.mean(dim=1).reshape(n, -1)
        assert torch.allclose(a.embed(x), want, atol=1e-6)


def test_timesfm_fully_padded_patches_excluded():
    a = _adapter()
    rows = torch.randn(2, 64)
    padded, mask = a._backbone.pad(rows)
    # emulate extra fully padded leading patches: embed must equal that of the unpadded series
    extra = torch.cat([torch.zeros(2, 64), padded], dim=-1)
    emask = torch.cat([torch.ones(2, 64, dtype=torch.bool), mask], dim=-1)
    m = a._model
    with torch.no_grad():
        h_ref, v_ref = a._backbone.tokens(rows)
        captured = []
        hk = m.stacked_xf[-1].register_forward_hook(lambda mod, i, o: captured.append(o[0]))
        m.decode(1, extra, emask)
        hk.remove()
    assert not captured[0].isnan().any()
    v = torch.cat([torch.zeros(2, 2, dtype=torch.bool), v_ref], dim=1)
    pooled = (captured[0] * v.unsqueeze(-1)).sum(1) / v.sum(1, keepdim=True)
    assert torch.allclose(pooled, h_ref.mean(1), atol=1e-4)  # padding does not change real tokens


def test_timesfm_channels_independent_and_nan_free():
    a = _adapter()
    x = a.preprocess(np.random.default_rng(0).standard_normal((2, 2, 64)))
    e = a.embed(x)
    assert tuple(e.shape) == (2, 2 * 1280) and not e.isnan().any()
    x2 = x.clone()
    x2[:, 1] = torch.randn(2, 64)
    assert torch.allclose(e[:, :1280], a.embed(x2)[:, :1280], atol=1e-4)
