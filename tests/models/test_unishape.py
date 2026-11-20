import numpy as np
import pytest
import torch

from tor.models import get_adapter
from tor.train import count_params

pytestmark = pytest.mark.slow

KEY_UNIT = "unit.convs.0.conv.weight"  # classifier-backbone name of their unit_scale_list.4.convs.0.conv.weight
KEY_QKV = "transformer_enc.transformer.layers.0.0.fn.to_qkv.weight"


def _adapter():
    a = get_adapter("unishape")()
    a.load("pretrained")
    return a


def test_unishape_availability_reports_reason(monkeypatch, tmp_path):
    import tor.models.unishape as U

    cls = get_adapter("unishape")
    assert cls.is_available() == (True, "")
    monkeypatch.setattr(U, "clone_dir", lambda: tmp_path / "nope")
    ok, why = cls.is_available()
    assert not ok and "clone missing" in why


def test_unishape_hash_mismatch_unavailable(monkeypatch):
    import tor.models.unishape as U

    monkeypatch.setattr(U, "CKPT_SHA256", "0" * 64)
    ok, why = get_adapter("unishape").is_available()
    assert not ok and "sha256 mismatch" in why


def test_unishape_stubs_not_left_in_sys_modules():
    import importlib.util
    import sys

    before = {n for n in ("timm", "fastai") if n in sys.modules}
    a = get_adapter("unishape")
    assert a.is_available()[0]
    a().load("pretrained")
    for n in ("timm", "fastai"):
        importlib.util.find_spec(n)  # must not raise ValueError (stub __spec__ None)
        assert (n in sys.modules) == (n in before)
    assert not any(getattr(m, "_tor_stub", False) for m in list(sys.modules.values()))


def test_unishape_pinned_commit_and_checkpoint_hash():
    import subprocess

    import tor.models.unishape as U

    d = U.clone_dir()
    assert subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() == U.COMMIT
    assert U._sha256(d / U.CKPT_REL) == U.CKPT_SHA256


def test_unishape_params_weights_scratch_lora():
    import tor.models.unishape as U

    a = _adapter()
    total, _ = count_params(a._model)
    unused = a.unused_param_count()
    parts = {n: sum(p.numel() for p in c.parameters()) for n, c in a._model.named_children()}
    print(f"[unishape] their fine-tuning model params (2-class fc): {total}; unused by classifier: {unused}; parts: {parts}")
    assert total == 3_133_203  # vs paper's 3.1M (Table 1): matches the full fine-tuning model, see below
    # the gap to the classifier's backbone is measured subtraction of exactly the unused parts
    expect_unused = (
        parts["unit_scale_list_finetune"]  # all 5 finetune units unused at scale 5
        + parts["fc"] + parts["fc_token_shape"] + parts["layer_norm"]
        + 128 * 2  # class_proto_centers (2 classes x 128)
        + 128  # transformer_enc.cls_token
        + (parts["unit_scale_list"] - sum(p.numel() for p in a._model.unit_scale_list[4].parameters()))
    )
    assert unused == expect_unused

    # weights really loaded from the checkpoint file (raw tensors, not via our remapping)
    raw = torch.load(U.clone_dir() / U.CKPT_REL, map_location="cpu", weights_only=False)["state_dict"]
    full = a._model_for(100)
    assert torch.equal(full.unit_scale_list[4].convs[0].conv.weight, raw["backbone.unit_scale_list.4.convs.0.conv.weight"])
    clf = a.classifier(3, "full", 1, 100)
    got = dict(clf.backbone.named_parameters())
    assert torch.equal(got[KEY_QKV], raw["backbone.transformer_enc.transformer.layers.0.0.fn.to_qkv.weight"])
    assert torch.equal(got[KEY_UNIT], raw["backbone.unit_scale_list.4.convs.0.conv.weight"])
    assert torch.equal(got["inceptime_token.0.convs.0.weight"], raw["backbone.inceptime_token.0.convs.0.weight"])

    a.load(("random", 0))
    rnd = a._model_for(100).state_dict()["transformer_enc.transformer.layers.0.0.fn.to_qkv.weight"].clone()
    assert not torch.equal(rnd, got[KEY_QKV])
    a.load(("random", 0))
    assert torch.equal(rnd, a._model_for(100).state_dict()["transformer_enc.transformer.layers.0.0.fn.to_qkv.weight"])
    a.load(("random", 1))
    assert not torch.equal(rnd, a._model_for(100).state_dict()["transformer_enc.transformer.layers.0.0.fn.to_qkv.weight"])
    s = a.classifier(3, "scratch", 1, 100, seed=0).backbone.state_dict()[KEY_QKV]
    assert not torch.equal(s, got[KEY_QKV])

    a.load("pretrained")
    counts = {m: count_params(a.classifier(3, m, 1, 100))[1] for m in ("head", "lora", "full")}
    print(f"[unishape] trainable params (3 classes, 1 channel): {counts}")
    head = 128 * 256 + 256 + 2 * 256 + 256 * 3 + 3
    assert counts["head"] == head
    assert counts["full"] == total - unused + head
    assert counts["head"] < counts["lora"] < counts["full"]
    ftot, ftr = count_params(a.classifier(3, "full", 1, 100))
    assert ftot == ftr
    assert count_params(a.classifier(3, "head", 1, 100).backbone)[1] == 0
    lm = a.classifier(3, "lora", 1, 100)
    names = [n for n, p in lm.backbone.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n and (".to_qkv." in n or ".to_out.0." in n) for n in names)
    assert len(names) == 6 * 2 * 2
    # same seed reproduces the head, different seeds differ
    h0 = a.classifier(3, "head", 1, 100, seed=0).head[0].weight
    assert torch.equal(h0, a.classifier(3, "head", 1, 100, seed=0).head[0].weight)
    assert not torch.equal(h0, a.classifier(3, "head", 1, 100, seed=1).head[0].weight)


def test_unishape_preprocess():
    a = get_adapter("unishape")()
    x = a.preprocess(np.random.default_rng(0).standard_normal((2, 3, 700)))
    assert tuple(x.shape) == (2, 3, 700) and x.dtype == torch.float32  # native length, no truncation
    assert tuple(a.preprocess(np.zeros((2, 5))).shape) == (2, 1, 16)  # short series resized, 2-D input gets a channel
    bad = np.zeros((2, 1, 64))
    bad[0, 0, 3] = np.inf
    with pytest.raises(ValueError):
        a.preprocess(bad)


@pytest.mark.parametrize("length", [100, 40])  # stride 4 and stride 2 branches of their constructor
def test_unishape_hidden_states_equal_their_forward(length):
    a = _adapter()
    x = a.preprocess(np.random.default_rng(0).standard_normal((3, 1, length)))
    theirs = a._model_for(length).eval()  # their UniShapeModel, same weights
    cap = []
    h = theirs.transformer_enc.register_forward_hook(lambda m, i, o: cap.append(o[0]))
    with torch.no_grad():
        theirs(x, torch.tensor([0, 1, 0]))  # their forward (needs labels for its losses)
    h.remove()
    assert torch.equal(a.embed(x), cap[0])  # exact: adapter class token == their transformer_enc class token


def test_unishape_channels_independent_shapes_and_finite():
    a = _adapter()
    x = a.preprocess(np.random.default_rng(0).standard_normal((2, 3, 64)))
    e = a.embed(x)
    assert tuple(e.shape) == (2, 3 * 128) and torch.isfinite(e).all()
    x2 = x.clone()
    x2[:, 2] = torch.randn(2, 64)
    e2 = a.embed(x2)
    assert torch.allclose(e[:, :256], e2[:, :256], atol=1e-5) and not torch.allclose(e[:, 256:], e2[:, 256:])


def test_unishape_short_and_constant_inputs_finite():
    a = _adapter()
    short = a.preprocess(np.random.default_rng(0).standard_normal((2, 1, 5)))
    const = torch.full((2, 1, 64), 3.0)
    for x in (short, const, torch.zeros(2, 1, 64)):
        e = a.embed(x)
        assert tuple(e.shape) == (2, 128) and torch.isfinite(e).all()
