import numpy as np
import pytest
import torch

from tor.models import get_adapter
from tor.train import count_params

pytestmark = pytest.mark.slow

# (name, d_model); parameter bounds are the paper's 40M / 125M / 385M (arXiv 2402.03885 Tab. 8)
CASES = [("moment-small", 512, 30e6, 45e6), ("moment-base", 768, 100e6, 140e6),
         ("moment-large", 1024, 300e6, 420e6)]


@pytest.mark.parametrize("name,d,lo,hi", CASES)
def test_moment_params_scratch_lora(name, d, lo, hi):
    a = get_adapter(name)()
    a.load("pretrained")
    total, _ = count_params(a._moment)
    unused = a._moment.encoder.embed_tokens.weight.numel()
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


def test_moment_padding_mask_matters_and_embed_shape():
    a = get_adapter("moment-small")()
    a.load("pretrained")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2, 3, 64)).astype("float32")
    e = a.embed(a.preprocess(X))
    assert tuple(e.shape) == (2, 3 * 512)
    # pad value must not leak: NaN pad and the mask give a result independent of other series
    e1 = a.embed(a.preprocess(X[:1]))
    assert torch.allclose(e[:1], e1, atol=1e-4)
    a._backbone.train()
    a.embed(a.preprocess(X))
    assert not a._backbone.training
