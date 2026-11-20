import numpy as np
import pytest
import torch

from tor.models import get_adapter
from tor.train import count_params

pytestmark = pytest.mark.slow

KEY_QKV = "wint.transformer.layers.0.fn.to_qkv.weight"  # classifier-backbone name of checkpoint backbone.1.transformer.layers.0.fn.to_qkv.weight


def _adapter():
    a = get_adapter("nutime")()
    a.load("pretrained")
    return a


def _raw():
    import tor.models.nutime as N

    assert N._sha256(N.clone_dir() / N.CKPT_REL) == N.CKPT_SHA256
    return torch.load(N.clone_dir() / N.CKPT_REL, map_location="cpu", weights_only=False)["state_dict"]


def test_nutime_availability_reports_reason(monkeypatch, tmp_path):
    import tor.models.nutime as N

    cls = get_adapter("nutime")
    assert cls.is_available() == (True, "")
    monkeypatch.setattr(N, "clone_dir", lambda: tmp_path / "nope")
    ok, why = cls.is_available()
    assert not ok and "clone missing" in why


def test_nutime_hash_mismatch_unavailable(monkeypatch):
    import tor.models.nutime as N

    monkeypatch.setattr(N, "CKPT_SHA256", "0" * 64)
    ok, why = get_adapter("nutime").is_available()
    assert not ok and "sha256 mismatch" in why


def test_nutime_no_sys_modules_leakage():
    import importlib.util
    import sys

    before = {n for n in sys.modules if n == "src" or n.startswith("src.") or n == "timm" or n.startswith("timm.")}
    a = get_adapter("nutime")
    assert a.is_available()[0]
    a().load("pretrained")
    after = {n for n in sys.modules if n == "src" or n.startswith("src.") or n == "timm" or n.startswith("timm.")}
    assert after == before
    importlib.util.find_spec("timm")  # must not raise ValueError (stub __spec__ None)
    assert not any(getattr(m, "_tor_stub", False) for m in list(sys.modules.values()))


def test_nutime_pinned_commit_and_checkpoint_hash():
    import subprocess

    import tor.models.nutime as N

    d = N.clone_dir()
    assert subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() == N.COMMIT
    assert N._sha256(d / N.CKPT_REL) == N.CKPT_SHA256


def test_nutime_params_weights_scratch_lora():
    a = _adapter()
    raw = _raw()
    total, _ = count_params(a._model)
    unused = a.unused_param_count()
    parts = {n: sum(p.numel() for p in c.parameters()) for n, c in a._model.named_children()}
    ck_total = sum(v.numel() for k, v in raw.items() if not k.endswith(("num_batches_tracked",)) and "running_" not in k)
    print(f"[nutime] their fine-tuning model params (2-class fc): {total}; unused by classifier: {unused}; parts: {parts}")
    print(f"[nutime] checkpoint tensors (incl. BYOL momentum copy, predictor, projector, buffers): {ck_total}")
    assert total == sum(p.numel() for p in a._model.parameters())
    assert unused == 128 * 2 + 2  # only their fc (2 classes) is not used by the embedding path
    assert total - unused == sum(p.numel() for p in a.classifier(2, "full", 1, 512).backbone.parameters())

    # weights really loaded from the checkpoint file (raw tensors, not via our remapping), strict
    clf = a.classifier(3, "full", 1, 512)
    got = dict(clf.backbone.named_parameters())
    assert torch.equal(got[KEY_QKV], raw["backbone.1.transformer.layers.0.fn.to_qkv.weight"])
    assert torch.equal(got["wint.cls_token"], raw["backbone.1.cls_token"])
    assert torch.equal(got["encoder.window_embedding.weight"], raw["backbone.0.window_embedding.weight"])
    assert torch.equal(got["encoder.multi_bias_encoder.ws"], raw["backbone.0.multi_bias_encoder.ws"])
    assert torch.equal(clf.backbone.wint.pos_embed, raw["backbone.1.pos_embed"])  # fixed sinusoid buffer == checkpoint's
    assert torch.equal(clf.backbone.encoder.multi_bias_encoder.scales, raw["backbone.0.multi_bias_encoder.scales"])

    a.load(("random", 0))
    rnd = a.classifier(3, "scratch", 1, 512, seed=0).backbone.state_dict()[KEY_QKV].clone()
    assert not torch.equal(rnd, got[KEY_QKV])
    assert torch.equal(rnd, a.classifier(3, "scratch", 1, 512, seed=0).backbone.state_dict()[KEY_QKV])
    assert not torch.equal(rnd, a.classifier(3, "scratch", 1, 512, seed=1).backbone.state_dict()[KEY_QKV])
    assert not torch.equal(a._model.state_dict()["1.transformer.layers.0.fn.to_qkv.weight"], got[KEY_QKV])

    a.load("pretrained")
    counts = {m: count_params(a.classifier(3, m, 1, 512))[1] for m in ("head", "lora", "full")}
    print(f"[nutime] trainable params (3 classes, 1 channel): {counts}")
    head = 128 * 3 + 3
    assert counts["head"] == head
    assert counts["full"] == total - unused + head
    assert counts["head"] < counts["lora"] < counts["full"]
    ftot, ftr = count_params(a.classifier(3, "full", 1, 512))
    assert ftot == ftr
    assert count_params(a.classifier(3, "head", 1, 512).backbone)[1] == 0
    lm = a.classifier(3, "lora", 1, 512)
    names = [n for n, p in lm.backbone.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n and (".to_qkv." in n or ".to_out.0." in n) for n in names)
    assert len(names) == 6 * 2 * 2
    h0 = a.classifier(3, "head", 1, 512, seed=0).head.weight
    assert torch.equal(h0, a.classifier(3, "head", 1, 512, seed=0).head.weight)
    assert not torch.equal(h0, a.classifier(3, "head", 1, 512, seed=1).head.weight)


def test_nutime_preprocess():
    a = get_adapter("nutime")()
    x = a.preprocess(np.random.default_rng(0).standard_normal((2, 3, 700)))
    assert tuple(x.shape) == (2, 3, 512) and x.dtype == torch.float32  # resized to transform_size 512
    assert tuple(a.preprocess(np.zeros((2, 5))).shape) == (2, 1, 512)  # 2-D input gets a channel, short is upsampled
    bad = np.zeros((2, 1, 64))
    bad[0, 0, 3] = np.nan
    with pytest.raises(ValueError):
        a.preprocess(bad)
    bad[0, 0, 3] = np.inf
    with pytest.raises(ValueError):
        a.preprocess(bad)


def test_nutime_resize_equals_their_TSResize():
    import tor.models.nutime as N

    tr = N._their_transforms()
    raw = torch.randn(3, 1, 178)
    ours = get_adapter("nutime")().preprocess(raw.numpy())
    theirs = tr.TSResize(size=512, mode="linear")
    for i in range(3):
        assert torch.equal(ours[i], theirs(raw[i]))


@pytest.mark.parametrize("length", [512, 176])
def test_nutime_hidden_states_equal_their_forward(length):
    """Adapter embedding == `features` (input of fc) of their nn.Sequential(encoder, WinT) built as in cls_exp.py."""
    import tor.models.nutime as N

    a = _adapter()
    x = torch.randn(3, 1, length)  # their model takes any length that is a multiple of 16 (here 512 / 176)
    theirs = a._their_model(length).eval()
    with torch.no_grad():
        theirs(x)
    feats = theirs[1].features
    emb = a.embed(x)
    assert torch.equal(emb, feats)


def test_nutime_channels_independent_shapes_and_finite():
    a = _adapter()
    x = a.preprocess(np.random.default_rng(0).standard_normal((2, 3, 64)))
    e = a.embed(x)
    assert tuple(e.shape) == (2, 3 * 128) and torch.isfinite(e).all()
    x2 = x.clone()
    x2[:, 2] = torch.randn(2, 512)
    e2 = a.embed(x2)
    assert torch.allclose(e[:, :256], e2[:, :256], atol=1e-5) and not torch.allclose(e[:, 256:], e2[:, 256:])


def test_nutime_short_and_constant_inputs_finite():
    a = _adapter()
    short = a.preprocess(np.random.default_rng(0).standard_normal((2, 1, 5)))
    for x in (short, torch.full((2, 1, 512), 3.0), torch.zeros(2, 1, 512), torch.full((2, 1, 512), 1e6)):
        e = a.embed(x)
        assert tuple(e.shape) == (2, 128) and torch.isfinite(e).all()
