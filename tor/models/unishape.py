"""UniShape adapter: `unishape` (Liu et al., "A Unified Shape-Aware Foundation Model for Time Series
Classification", arXiv 2601.06429, AAAI 2026; code github.com/qianlima-lab/UniShape).

UniShape code/weights have no licence file; used for research measurement only; inclusion in the study is a user decision.

Nothing of UniShape is vendored in this repository. This file is our own code that imports their modules from a
clone of their repository kept in the cache directory (`<TOR_CACHE>/unishape/UniShape`), pinned to commit
`78d326590ee959bd473c4b07ad6cbc13977c8625` (2025-11-13). Set up with:
    git clone https://github.com/qianlima-lab/UniShape.git <TOR_CACHE>/unishape/UniShape
    git -C <TOR_CACHE>/unishape/UniShape checkout 78d326590ee959bd473c4b07ad6cbc13977c8625
The checkpoint is `pretrained_model_ckpt/unishape_checkpoint_finetune.pth` of that clone (no HF repo), sha256
`1a908b6bf9b7cd1b90c393514d069cb5e9f018cd2af189727a244f1e41eb65df`, checked before loading (the file is a pickle).
`is_available()` returns (False, reason) if the clone is missing, not at the pinned commit, or the checkpoint is
missing / has another hash.

Imports: their package is loaded as `_unishape_models` (importlib, `models/` directory) so no generic top-level
name (`models`, `utils`) is put on sys.path. `models/networks.py` does `from fastai.basics import *` but uses only
`Module`, `noop`, `init_linear`; fastai's dependency tree (matplotlib, fasthtml, ...) is not installed, so a
minimal stand-in `fastai.basics` with those three names (behaviour copied from fastai's definitions, as
remembered; not diffed against fastai) is registered in sys.modules unless real fastai imports. `TransformerEnc`
calls `torch.randn(...).cuda()` in its constructor, so `Tensor.cuda` is a no-op while their model is built (no GPU
is touched). Their code needs `einops` (installed) and `timm.models.layers.trunc_normal_`: timm imports torchvision (not
installed), so the same function (`torch.nn.init.trunc_normal_`; identity with timm's is inferred, not verified) is stubbed in.

Reference path: `unishape_finetune.py` + `models/unishapemodel_finetune.py::UniShapeModel` (their classification
fine-tuning model).
* Length: the fine-tuning script does NOT resize. `series_size = train_x.shape[1]` (native length); the model is
  built per series length. Resizing to 512 (`datapre/transforms.py::TSResize`, linear `F.interpolate`, via
  `datapre/dataloader.py`, `transform_size=512` in `unishape_pretrain.py`) is used for pretraining and
  `unishape_zeroshot.py`, not fine-tuning. So `max_length=None` (no truncation; positions are sinusoidal and
  sliced to the token count). `preprocess` only rejects NaN/inf (their script instead fills NaN with train column
  means, `datapre/preprocessing.py::fill_nan_value`) and linearly resizes series shorter than 16 points to 16 (harness
  choice with their TSResize op; their code needs >= 6 points for >= 2 tokens).
* Scale: `UniShapeModel.forward` uses ONE window scale, `window_size_list[scale_len-1]` of [64,32,16,8,4];
  per-dataset `scale_len` for the 128 UCR sets is in `utils/hyp_batch_shape_sizes.py`. The harness has no such
  table, so it uses the script's argparse default `--scale_len 5` (window 4, stride 2 if series < 64 else 4, as
  in the model constructor). The pretrained units for scale 5 are `unit_scale_list[4]` (as in forward).
* Channels: their script is univariate only (`train_x[:, :, np.newaxis]`; UEA data were split into univariate
  series for pretraining, per the paper). We run channels independently (per_channel) and concatenate: d = 128*C.
* Embedding: the class token returned by `TransformerEnc` (the input to their `fc` head): scale tokens ->
  InceptionTime tokens (`inceptime_token`) -> LayerNorm/dropout/GELU -> `attention_head` weights -> mean = class
  token seed -> 6-layer Transformer encoder (dim 128, 8 heads, mlp 512) -> output class token. Their own pooling,
  no mean pooling over tokens. `_Backbone.forward` re-implements those lines of `UniShapeModel.forward` on their
  submodules (their forward needs labels and computes losses); test_unishape.py asserts torch.equal against
  their forward's `transformer_enc` output.
* Head: their `fc` = Linear(128,256) - BatchNorm1d - ReLU - Dropout(0.1) - Linear(256,K), initialised with
  trunc_normal(std .02) (their `_init_weights`; the checkpoint's fc is discarded by `load_pretrained_model`, so
  the head is always fresh). Input size 128*C here. Their auxiliary prototype loss (weight 0.01) is not used:
  harness trains with cross-entropy only. Classifiers delete everything their forward does not use for the
  embedding: unit_scale_list[0..3] and all unit_scale_list_finetune, `fc_token_shape`, `class_proto_centers`,
  `layer_norm`, `fc`, `transformer_enc.cls_token`; `unused_param_count()` reports them.
* Scratch: `UniShapeModel(...)` constructed under a seeded RNG (their `_init_weights`: trunc_normal Linear,
  LayerNorm 1/0; other layers PyTorch/fastai defaults), no pretrained weights.
* LoRA: peft on `to_qkv` (fused q,k,v Linear) and `to_out.0` (output Linear inside nn.Sequential) of each of the 6
  Transformer layers, `transformer_enc.transformer.layers.N.0.fn.*`. Head trainable.
* Gradient checkpointing: torch.utils.checkpoint on each PreNorm attention/MLP block (their code has none),
  training mode only.
* Recipe (`default_recipe`): `unishape_finetune.py`: AdamW, lr 1e-3, weight_decay 0.05, `--epoch 300` (matches
  R2R's "up to 300 epochs" for UniShape), batch size argparse default 512 (overridden per dataset from the
  hyp table; modal value 512), constant LR, no warmup. Script differences not reproduced: 5-model ensemble,
  prototype loss, checkpoint selection by train loss.
* Pretraining corpora (paper, arXiv 2601.06429 as read from notes lit-c.md and the repo README): 1.89M samples
  from the UCR (128) and UEA (30) archives' TRAIN splits plus eight TFC datasets; UCR/UEA data ARE included
  (test splits excluded per the paper).
* Params: see test_unishape.py (measured, with subtraction of the unused parts vs the paper's 3.1M).
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib
import subprocess
import sys
import types
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from tor.models import register
from tor.models.base import AdapterUnavailable, FMAdapter, Mode, Recipe

COMMIT = "78d326590ee959bd473c4b07ad6cbc13977c8625"
CKPT_SHA256 = "1a908b6bf9b7cd1b90c393514d069cb5e9f018cd2af189727a244f1e41eb65df"
CKPT_REL = "pretrained_model_ckpt/unishape_checkpoint_finetune.pth"
_D = 128  # in_channels / window_emb_dim in unishape_finetune.py
_SCALE_LEN = 5  # argparse default of unishape_finetune.py
_UNIT = _SCALE_LEN - 1
_MIN_LEN = 16


def clone_dir():
    from tor.config import get_settings

    return get_settings().cache_dir / "unishape" / "UniShape"


@functools.lru_cache(maxsize=None)
def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _check_clone() -> tuple[bool, str]:
    d = clone_dir()
    if not (d / "models" / "unishapemodel_finetune.py").exists():
        return False, f"UniShape clone missing at {d} (git clone github.com/qianlima-lab/UniShape, checkout {COMMIT})"
    try:
        head = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        return False, f"cannot read UniShape clone commit: {e}"
    if head != COMMIT:
        return False, f"UniShape clone at {head}, expected pinned {COMMIT}"
    if not (d / CKPT_REL).exists():
        return False, f"UniShape checkpoint missing: {d / CKPT_REL}"
    if _sha256(d / CKPT_REL) != CKPT_SHA256:  # hashed once per process (56 MB)
        return False, f"UniShape checkpoint sha256 mismatch: {d / CKPT_REL}"
    return True, ""


def _install_timm_stub() -> None:
    """Their model imports `timm.models.layers.trunc_normal_`; timm needs torchvision, so provide the same
    function (inferred, not diffed since timm is not installed: timm's `trunc_normal_(tensor, mean=0., std=1.,
    a=-2., b=2.)` is torch's) unless timm imports. The stubs are removed from sys.modules right after the import."""
    try:
        from timm.models.layers import trunc_normal_  # noqa: F401
        return
    except ImportError:
        pass
    for name in ("timm", "timm.models", "timm.models.layers"):
        sys.modules[name] = types.ModuleType(name)
        sys.modules[name].__path__ = []
    sys.modules["timm.models.layers"].trunc_normal_ = nn.init.trunc_normal_
    for name in ("timm", "timm.models", "timm.models.layers"):
        sys.modules[name]._tor_stub = True


def _install_fastai_stub() -> None:
    try:
        import fastai.basics  # noqa: F401
        return
    except ImportError:
        pass

    class _Meta(type(nn.Module)):  # fastai PrePostInitMeta: nn.Module.__init__ runs before the subclass __init__
        def __call__(cls, *a, **k):
            obj = cls.__new__(cls)
            nn.Module.__init__(obj)
            obj.__init__(*a, **k)
            return obj

    class Module(nn.Module, metaclass=_Meta):
        def __init__(self):
            pass

    def noop(x=None, *a, **k):
        return x

    def init_linear(m, act_func=None, init="auto", bias_std=0.01):
        if getattr(m, "bias", None) is not None and bias_std is not None:
            if bias_std != 0:
                nn.init.normal_(m.bias, 0, bias_std)
            else:
                m.bias.data.zero_()
        if init == "auto":
            init = getattr(act_func.__class__, "__default_init__", None) or getattr(act_func, "__default_init__", None)
        if init is not None:
            init(m.weight)

    pkg = types.ModuleType("fastai")
    pkg.__path__ = []
    mod = types.ModuleType("fastai.basics")
    mod.Module, mod.noop, mod.init_linear = Module, noop, init_linear
    mod.__all__ = ["Module", "noop", "init_linear"]
    pkg.basics = mod
    pkg._tor_stub = mod._tor_stub = True
    sys.modules["fastai"], sys.modules["fastai.basics"] = pkg, mod


_PKG = "_unishape_models"


def _their_module():
    """Import `models/unishapemodel_finetune.py` of the pinned clone as `_unishape_models.unishapemodel_finetune`."""
    ok, why = _check_clone()
    if not ok:
        raise AdapterUnavailable(why)
    name = f"{_PKG}.unishapemodel_finetune"
    if name in sys.modules:
        return sys.modules[name]
    _install_fastai_stub()
    _install_timm_stub()
    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [str(clone_dir() / "models")]
    sys.modules[_PKG] = pkg
    try:
        return importlib.import_module(name)
    except ImportError as e:
        del sys.modules[_PKG]
        raise AdapterUnavailable(f"UniShape imports failed: {e}") from e
    finally:  # stubs have __spec__ None and would break find_spec("timm"/"fastai"); their modules keep references
        for n in [n for n, m in list(sys.modules.items()) if getattr(m, "_tor_stub", False)]:
            del sys.modules[n]


try:
    import einops  # noqa: F401
except ImportError as e:  # registry skips the module
    raise AdapterUnavailable(f"unishape needs einops: {e}") from e


@contextlib.contextmanager
def _no_cuda_tensor():
    """Their `TransformerEnc.__init__` calls `torch.randn(...).cuda()`; keep construction on the CPU."""
    orig = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *a, **k: self
    try:
        yield
    finally:
        torch.Tensor.cuda = orig


def build_their_model(series_size: int, n_out: int = 2) -> nn.Module:
    """Their `UniShapeModel` exactly as built in unishape_finetune.py (random init; fine-tuning configuration)."""
    mod = _their_module()
    cfg = SimpleNamespace(window_size=16, stride=16)
    with _no_cuda_tensor():
        return mod.UniShapeModel(
            config=cfg, series_size=series_size, in_channels=_D, window_emb_dim=_D, out_channels=n_out,
            window_size=16, stride=16, pre_training=False, shape_alpha=0.01, shape_ratio=0.6, scale_len=_SCALE_LEN,
        )


def checkpoint_state_dict() -> dict:
    """Checkpoint state, remapped like `utils/util.py::load_pretrained_model` (strip `backbone.`, drop fc/pe)."""
    path = clone_dir() / CKPT_REL
    if _sha256(path) != CKPT_SHA256:
        raise AdapterUnavailable(f"UniShape checkpoint sha256 mismatch: {path}")
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]  # hash-verified pickle
    return {k[len("backbone."):]: v for k, v in sd.items()
            if k.startswith("backbone.") and not k.startswith("backbone.fc.")
            and k != "backbone.transformer_enc.pos_encoder.pe"}


_ALLOWED_MISSING = ("fc.", "class_proto_centers", "transformer_enc.pos_encoder.pe")


def load_pretrained_into(model: nn.Module, sd: dict) -> None:
    """strict load except for the parts their loader also skips (fresh head, class centres, sinusoidal pe)."""
    sd = {k: v for k, v in sd.items() if k != "class_proto_centers"}  # (n_classes,128): shape depends on data
    res = model.load_state_dict(sd, strict=False)
    bad = [k for k in res.missing_keys if not k.startswith(_ALLOWED_MISSING)]
    if bad or res.unexpected_keys:
        raise RuntimeError(f"UniShape checkpoint mismatch: missing {bad}, unexpected {res.unexpected_keys}")


def _wrap_checkpoint(block: nn.Module) -> None:
    orig = block.forward

    def fwd(*a, **k):
        if block.training and torch.is_grad_enabled():
            return checkpoint(orig, *a, use_reentrant=False, **k)
        return orig(*a, **k)

    block.forward = fwd


class _Backbone(nn.Module):
    """(n, C, L) -> (n, C*128): their class token per channel (see module docstring)."""

    def __init__(self, full: nn.Module):
        super().__init__()
        self.unit = full.unit_scale_list[_UNIT]
        self.inceptime_token = full.inceptime_token
        self.layer_norm_inc = full.layer_norm_inc
        self.drop_token = full.drop_token
        self.act = full.act_gelu_inc
        self.attention_head = full.attention_head
        self.transformer_enc = full.transformer_enc
        del self.transformer_enc.cls_token  # unused: forward passes `cls_token_in`

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, length = x.shape
        x = x.reshape(n * c, 1, length)
        x_embed = self.unit(x)  # UniShapeModel.forward, scale_len == 5 branch
        t = self.inceptime_token(x_embed.permute(0, 2, 1)).permute(0, 2, 1)
        t = self.act(self.drop_token(self.layer_norm_inc(t)))
        cls_seed = (t * self.attention_head(t)).mean(dim=1).unsqueeze(1)
        cls_token, _ = self.transformer_enc(x_embed, cls_token_in=cls_seed)
        return cls_token.reshape(n, c * _D)


class _Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, n_classes: int, n_channels: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(  # their `fc`
            nn.Linear(_D * n_channels, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, n_classes),
        )
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


@register
class UniShape(FMAdapter):
    name = "unishape"
    hf_repo = None  # GitHub-hosted checkpoint (no HF repo)
    max_length = None
    multivariate = "per_channel"
    supported_modes: tuple[Mode, ...] = ("head", "lora", "full", "scratch")
    supports_grad_ckpt = True
    lora_targets = ("to_qkv", "to_out.0")
    pretraining_corpora = (
        "UniShape paper (Liu et al., arXiv 2601.06429, AAAI 2026): 1.89M univariate samples built from the TRAIN "
        "splits of the UCR archive (128 datasets) and UEA archive (30 datasets, split into univariate series) "
        "plus eight TFC datasets; UCR/UEA data ARE included (test splits excluded per the paper; overlap with "
        "downstream UCR train sets not discussed)."
    )
    recipe_source = (
        "unishape_finetune.py of github.com/qianlima-lab/UniShape @ " + COMMIT[:10] + ": AdamW lr 1e-3, "
        "weight_decay 0.05, --epoch 300, --batch_size 512 default (per-dataset overrides in "
        "utils/hyp_batch_shape_sizes.py), constant LR; matches R2R's 'up to 300 epochs' for UniShape"
    )

    def __init__(self):
        self._init_kind = None
        self._seed = 0
        self._sd = None
        self._cache: dict[int, _Backbone] = {}

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        ok, why = _check_clone()
        if not ok:
            return False, why
        try:
            _their_module()
        except AdapterUnavailable as e:
            return False, str(e)
        return True, ""

    @staticmethod
    def _build(seed: int, series_size: int, n_out: int = 2) -> nn.Module:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return build_their_model(series_size, n_out)

    def load(self, init) -> None:
        self._cache = {}
        if init == "pretrained":
            self._sd = checkpoint_state_dict()
            self._init_kind = "pretrained"
        else:
            kind, seed = init
            if kind != "random":
                raise ValueError(f"bad init {init!r}")
            self._sd, self._seed, self._init_kind = None, seed, "random"
        self._model = self._model_for(512)  # reference model (params are independent of series length)

    def _model_for(self, length: int, seed: int | None = None) -> nn.Module:
        m = self._build(self._seed if seed is None else seed, length)
        if self._init_kind == "pretrained":
            load_pretrained_into(m, self._sd)
        return m

    def unused_param_count(self) -> int:
        """Params of the loaded their-model (2-class `fc`) that the classifier's embedding path does not use."""
        if self._init_kind is None:
            raise RuntimeError("call load() first")
        total = sum(p.numel() for p in self._model.parameters())
        return total - sum(p.numel() for p in _Backbone(self._build(0, 512)).parameters())

    def preprocess(self, X: np.ndarray) -> torch.Tensor:
        x = np.asarray(X, dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError(f"{self.name}: input contains NaN/inf; impute first")
        if x.ndim == 2:
            x = x[:, None, :]
        t = torch.as_tensor(np.ascontiguousarray(x))
        if t.shape[-1] < _MIN_LEN:  # linear resize as their TSResize (F.interpolate, align_corners=False)
            t = F.interpolate(t, size=_MIN_LEN, mode="linear", align_corners=False)
        return t

    def embed(self, X: torch.Tensor) -> torch.Tensor:
        if self._init_kind is None:
            raise RuntimeError("call load() first")
        length = X.shape[-1]
        if length not in self._cache:
            self._cache[length] = _Backbone(self._model_for(length)).eval()
        with torch.no_grad():
            return self._cache[length](X)

    def classifier(self, n_classes, mode, n_channels, seq_len, grad_ckpt=False, seed=0):
        self._check_mode(mode, grad_ckpt)
        if mode == "scratch":
            full = self._build(seed, seq_len)
        else:
            if self._init_kind != "pretrained":
                raise RuntimeError(f"{mode} needs load('pretrained') first")
            full = self._build(seed, seq_len)
            load_pretrained_into(full, self._sd)
        backbone = _Backbone(full)
        for p in backbone.parameters():
            p.requires_grad = True
        if grad_ckpt:
            for pair in backbone.transformer_enc.transformer.layers:
                for blk in pair:
                    _wrap_checkpoint(blk)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if mode == "lora":
                import peft

                backbone = peft.get_peft_model(backbone, self.lora_config())
            elif mode == "head":
                for p in backbone.parameters():
                    p.requires_grad = False
            return _Classifier(backbone, n_classes, n_channels)

    def default_recipe(self) -> Recipe:
        return Recipe(lr=1e-3, weight_decay=0.05, epochs=300, batch_size=512)
