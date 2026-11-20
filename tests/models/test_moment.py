import numpy as np
import pytest
import torch

from tor.models import get_adapter
from tor.train import count_params

pytestmark = pytest.mark.slow

# (name, d_model, lo, hi); sanity bounds around the measured totals (35.3M / 109.6M / 341.2M)
CASES = [("moment-small", 512, 30e6, 45e6), ("moment-base", 768, 100e6, 140e6),
         ("moment-large", 1024, 300e6, 420e6)]


@pytest.mark.parametrize("name,d,lo,hi", CASES)
def test_moment_params_scratch_lora(name, d, lo, hi):
    a = get_adapter(name)()
    a.load("pretrained")
    total, _ = count_params(a._moment)
    unused = a.unused_param_count()
    print(f"[{name}] pretrained params (no head): {total}; of which unused T5 embed_tokens: {unused}")
    assert lo < total < hi

    key = "encoder.block.0.layer.0.SelfAttention.q.weight"
    pre = a._moment.state_dict()[key].clone()
    a.load(("random", 0))
    rnd = a._moment.state_dict()[key].clone()
    assert not torch.equal(pre, rnd)
    a.load(("random", 0))
    assert torch.equal(rnd, a._moment.state_dict()[key])  # same seed, same init
    a.load(("random", 1))
    assert not torch.equal(rnd, a._moment.state_dict()[key])

    a.load("pretrained")
    counts = {m: count_params(a.classifier(3, m, 1, 512))[1] for m in ("head", "lora", "full")}
    print(f"[{name}] trainable params (3 classes, 1 channel): {counts}")
    assert counts["head"] == d * 3 + 3
    assert counts["full"] == total - unused + counts["head"]
    full_tot, full_tr = count_params(a.classifier(3, "full", 1, 512))
    assert full_tot == full_tr
    assert counts["head"] < counts["lora"] < counts["full"]
    assert count_params(a.classifier(3, "head", 1, 512).backbone)[1] == 0


def test_moment_preprocess_pad_and_subsample():
    a = get_adapter("moment-small")()
    rng = np.random.default_rng(0)
    x = a.preprocess(rng.standard_normal((2, 3, 100)))
    assert tuple(x.shape) == (2, 3, 512) and x.dtype == torch.float32
    assert torch.isnan(x[:, :, :412]).all() and not torch.isnan(x[:, :, 412:]).any()  # left pad
    y = a.preprocess(rng.standard_normal((2, 700)))
    assert tuple(y.shape) == (2, 1, 512) and not torch.isnan(y).any()
    z = a.preprocess(np.arange(1024, dtype="float32")[None, None])
    assert z[0, 0, 0] == 0 and z[0, 0, -1] == 1023 and (z[0, 0, 1:] > z[0, 0, :-1]).all()


def test_moment_preprocess_rejects_nan():
    a = get_adapter("moment-small")()
    x = np.zeros((2, 1, 64))
    x[0, 0, 3] = np.nan
    with pytest.raises(ValueError):
        a.preprocess(x)


def test_moment_mask_is_used_and_embed_shape():
    a = get_adapter("moment-small")()
    a.load("pretrained")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2, 3, 64)).astype("float32")
    x = a.preprocess(X)
    e = a.embed(x)
    assert tuple(e.shape) == (2, 3 * 512)
    # same series zero-padded WITHOUT mask (no NaN) must differ from the masked embedding
    unmasked = x.clone()
    unmasked[torch.isnan(unmasked)] = 0.0
    assert not torch.allclose(e, a.embed(unmasked), atol=1e-3)
    # masked embedding of a series does not depend on the rest of the batch
    assert torch.allclose(e[:1], a.embed(a.preprocess(X[:1])), atol=1e-4)
    a._backbone.train()
    a.embed(x)
    assert not a._backbone.training
