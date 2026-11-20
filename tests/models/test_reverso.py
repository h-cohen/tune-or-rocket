import numpy as np
import pytest
import torch

from tor.models import get_adapter
from tor.train import count_params

pytestmark = pytest.mark.slow

# (adapter, checkpoint params incl. unused decoder, layers, d_model)
CASES = {"reverso": (2_609_585, 8, 128), "reverso-small": (550_161, 4, 64)}


def _adapter(name):
    a = get_adapter(name)()
    a.load("pretrained")
    return a


@pytest.mark.parametrize("name", list(CASES))
def test_reverso_params_weights_scratch_lora(name):
    from safetensors import safe_open

    from tor.models.reverso import _hub

    want_total, nlayers, d = CASES[name]
    a = _adapter(name)
    total, _ = count_params(a._model)
    unused = a.unused_param_count()
    print(f"[{name}] pretrained params (with forecast decoder): {total}; unused by classifier: {unused}; used: {total - unused}")
    assert total == want_total  # measured: README says 2.6M / 550K
    with safe_open(_hub(a.hf_repo, "model.safetensors"), "pt") as f:
        ck = f.get_tensor("layers.0.k")
        n_ck = sum(int(np.prod(f.get_slice(k).get_shape())) for k in f.keys())
    # checkpoint = parameters + the non-parameter sinusoidal PE buffer (reverso-base only)
    assert n_ck - total == (2096 * 128 if name == "reverso" else 0)
    pre = a._model.state_dict()["layers.0.k"].clone()
    assert torch.equal(pre, ck)  # weights really loaded

    a.load(("random", 0))
    rnd = a._model.state_dict()["layers.0.k"].clone()
    assert not torch.equal(pre, rnd)
    a.load(("random", 0))
    assert torch.equal(rnd, a._model.state_dict()["layers.0.k"])
    a.load(("random", 1))
    assert not torch.equal(rnd, a._model.state_dict()["layers.0.k"])

    a.load("pretrained")
    counts = {m: count_params(a.classifier(3, m, 1, 128))[1] for m in ("head", "lora", "full")}
    print(f"[{name}] trainable params (3 classes, 1 channel): {counts}")
    head = 2 * d * 2 + 2 * d * 3 + 3  # LayerNorm(2d) + Linear(2d, 3)
    assert counts["head"] == head
    assert counts["full"] == total - unused + head
    assert counts["head"] < counts["lora"] < counts["full"]
    assert count_params(a.classifier(3, "head", 1, 128).backbone)[1] == 0
    lm = a.classifier(3, "lora", 1, 128)
    names = [n for n, p in lm.backbone.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n and ".attention." in n and any(f".{t}." in n for t in ("q_proj", "k_proj", "v_proj", "o_proj")) for n in names)
    assert len(names) == (nlayers // 2) * 4 * 2


def test_reverso_preprocess():
    a = get_adapter("reverso-small")()
    x = a.preprocess(np.random.default_rng(0).standard_normal((2, 3, 100)))
    assert tuple(x.shape) == (2, 3, 100) and x.dtype == torch.float32  # no padding in preprocess
    z = a.preprocess(np.arange(2100, dtype="float32")[None, None])
    assert tuple(z.shape) == (1, 1, 2048) and z[0, 0, -1] == 2099 and z[0, 0, 0] == 52
    bad = np.zeros((2, 1, 64))
    bad[0, 0, 3] = np.inf
    with pytest.raises(ValueError):
        a.preprocess(bad)


@pytest.mark.parametrize("name", list(CASES))
def test_reverso_hidden_states_equal_package_forward(name):
    a = _adapter(name)
    x = a.preprocess(np.random.default_rng(0).standard_normal((2, 1, 2048)))
    rows = x[:, 0]
    caps = []
    hooks = [a._model.layers[i].register_forward_hook(lambda m, i_, o: caps.append(o)) for i in range(1, len(a._model.layers), 2)]
    with torch.no_grad():
        a._model(rows.unsqueeze(-1))  # the package's own Model.forward
        for h in hooks:
            h.remove()
        hs, mask = a._backbone.hidden_states(rows)
    assert len(caps) == len(hs) == CASES[name][1]  # one per (mixer+MLP) pair
    for c, h in zip(caps, hs):
        assert torch.equal(c, h)  # exact match with package
    assert mask.all()
    with torch.no_grad():
        want = torch.cat([h.mean(-1) for h in hs], dim=-1)
        assert torch.equal(a.embed(x), want)


def test_reverso_padding_and_pooling_real_positions_only():
    a = _adapter("reverso-small")
    x = a.preprocess(np.random.default_rng(0).standard_normal((2, 1, 100)))
    padded, mask = a._backbone.pad(x[:, 0])
    assert padded.shape == (2, 2048) and mask.sum(-1).tolist() == [100, 100]
    assert torch.equal(padded[:, :1948], x[:, 0, :1].expand(-1, 1948)) and torch.equal(padded[:, 1948:], x[:, 0])
    with torch.no_grad():
        hs, _ = a._backbone.hidden_states(x[:, 0])
        want = torch.cat([h[..., 1948:].mean(-1) for h in hs], dim=-1)
    assert torch.allclose(a.embed(x), want, atol=1e-6)
    # max pooling in the classifier backbone also covers real positions only
    from tor.models.reverso import _Backbone

    bb = _Backbone(a._model, "last").eval()
    with torch.no_grad():
        f = bb(x)
    assert torch.allclose(f[:, 64:], hs[-1][..., 1948:].amax(-1), atol=1e-6)


@pytest.mark.parametrize("name", list(CASES))
def test_reverso_channels_independent_short_constant_finite(name):
    a = _adapter(name)
    d = CASES[name][2] * CASES[name][1]  # d_model x one feature block per mixer+MLP pair
    x = a.preprocess(np.random.default_rng(0).standard_normal((2, 2, 64)))
    e = a.embed(x)
    assert tuple(e.shape) == (2, 2 * d) and torch.isfinite(e).all()
    x2 = x.clone()
    x2[:, 1] = torch.randn(2, 64)
    assert torch.allclose(e[:, :d], a.embed(x2)[:, :d], atol=1e-5)
    for z in (a.preprocess(np.random.default_rng(1).standard_normal((2, 1, 5))), torch.full((2, 1, 64), 3.0)):
        assert torch.isfinite(a.embed(z)).all()
