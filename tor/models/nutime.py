"""NuTime adapter: `nutime` (Lin, Wen, Cao, Huang, Bian, Lin, Wu, "NuTime: Numerically Multi-Scaled Embedding for
Large-Scale Time-Series Pretraining", arXiv 2310.07402, accepted by TMLR 2024 per the arXiv comments field; code
github.com/chenguolin/NuTime, MIT licence, Copyright (c) 2024 Chenguo Lin).

Nothing of NuTime is vendored. This file is our own code that imports their modules from a clone of their repository
kept in the cache directory (`<TOR_CACHE>/nutime/NuTime`), pinned to commit `9acdd3d0e82df0914217ef752fa3b95ec82cd960`
(see COMMIT; 2024-12-23). Set up with:
    git clone https://github.com/chenguolin/NuTime.git <TOR_CACHE>/nutime/NuTime
    git -C <TOR_CACHE>/nutime/NuTime checkout <COMMIT>
The checkpoint is `ckpt/checkpoint_bias9.pth` of that clone (no HF repo; BYOL self-supervised pretraining, 9 multi-scale
biases), sha256 `65543d55c6a9d30837b46b3809b7358a6ecb3e9702f4eaf01bfd68a96707f7a2`, checked before loading (the file is
a pickle). `is_available()` returns (False, reason) if the clone is missing, not at the pinned commit, or the
checkpoint is missing / has another hash.

Imports: their code uses absolute `src.models...` / `src.data...` imports, so while importing, packages named `src`,
`src.models`, `src.models.encoders`, `src.data` are registered in sys.modules pointing at the clone's directories (any
pre-existing `src*` modules are set aside and restored); afterwards all `src*` entries are removed again, the classes
(`WinT`, `WindowNormEncoder`, `TSResize`) stay referenced here. `timm` (needs torchvision, not installed) is used by
their code only for `trunc_normal_` and `DropPath`; `trunc_normal_` is stubbed with `torch.nn.init.trunc_normal_`
(only used by their learnable-mask / learnable-PE branches, not by this configuration) and `DropPath` with a class
that raises if instantiated (path_dropout is 0, so it is never built). Stubs are also removed after the import.

Reference path (fine-tuning): `configs/demo_ft_epilepsy.json` through `src/pipeline.py` ->
`src/experiments/cls_exp.py::ClassificationExp.build_model` (lines 14-30): `nn.Sequential(encoder, model)` with
`encoder = WindowNormEncoder` (`src/models/encoders/build.py:18-21`, config `encoder: wne`) and `model = WinT`
(`src/models/build.py:30-37`, `model: wint`), then `load_pretrained_model` (cls_exp.py:32-57).
* Length: the datasets pass every series through `TSResize(size=transform_size, mode=interpolate_mode)`
  (`src/data/dataloader.py:get_transform`; `TSResize = TSRandomResizedCrop(scale=[1,1])`, transforms.py:458;
  linear `F.interpolate`, align_corners=False, transforms.py:179) so ALL series are linearly resized (up or down) to
  one fixed length; `WindowNormEncoder` asserts length % window_size(16) == 0 (WindowNormEncoder.py:13).
  `transform_size` is 512 in their `train_ssl.json` (the pretraining config, dataset `ucr-merged`), `train_ft.json`,
  `train_cls.json`; the Epilepsy demo uses 176 (Epilepsy series have 178 points, so this is dataset-specific). We use
  512 = the pretraining/fine-tuning-template value: `max_length = 512`, `preprocess` resizes every series to 512
  (32 windows). Their `__getitem__` drops trailing all-NaN columns and zero-fills other NaN (dataset.py:101-104);
  we instead raise ValueError on NaN/inf (harness rule). `norm: identity` (no global normalisation) is kept: the
  window encoder normalises each window itself (WindowNormEncoder.py:71-86).
* Channels: pretraining data are univariate (the checkpoint has no `channel_embedding`; their multivariate path
  `WindowNormEncoder.channel_embedding`, a fresh Linear(C*128,128)+LayerNorm, is NOT pretrained). We run channels
  independently (per_channel) and concatenate: d = 128*C. This differs from their demo for multichannel data.
* Embedding: `WinT.features` = LayerNorm(cls token after the 6-layer Transformer) (networks.py:150-156), the input of
  their `fc` (pool_mode `cls`). The classifier's backbone is their encoder + WinT with `fc` replaced by `nn.Identity`,
  so WinT.forward itself computes the embedding; test_nutime.py asserts torch.equal with `features` of their
  `nn.Sequential(encoder, WinT)`.
* Head: their `fc` = Linear(128, K) (networks.py:70). Here Linear(128*C, K). `head` mode follows their
  `freeze_backbone` branch (cls_exp.py:59-72): weight ~ N(0, 0.01^2), bias 0; other modes use the PyTorch default init
  (as in their full fine-tuning). Their dropout config (0.1, attn_dropout 0.1) is kept in train mode.
* Scratch: same modules built (their code, `pretrained_model: false` path) under a seeded RNG; their init is the
  PyTorch default for Linear/Conv/LayerNorm, uniform(-1,1) for the multi-bias `ws`/`bs`, N(0,1e-6^2) cls token.
* LoRA: peft on `to_qkv` (fused q,k,v Linear, bias-free) and `to_out.0` (output Linear inside nn.Sequential) of
  the 6 attention layers, `wint.transformer.layers.{0,6,12,..}.fn.*`. Head trainable.
* Gradient checkpointing: torch.utils.checkpoint on each PreNorm attention/MLP block (their code has none), training
  mode only.
* Recipe (`default_recipe`): `configs/demo_ft_epilepsy.json`: AdamW (betas .9/.999), lr 2e-4, weight_decay 0.05,
  num_epochs 100 (max_epochs 500 and patience 64 are only the early-stopping settings), batch_size 2048, cosine
  schedule with 10 warmup epochs (warmup_frac 0.1; lr from 2e-6 to min_lr 2e-5). These are the demo's values, chosen for
  the Epilepsy dataset (large train set): batch 2048 exceeds most UCR train sets, i.e. one full-batch step per epoch.
  `train_ft.json` has lr 2e-3, batch 1024, 200 epochs instead; the paper (as read in notes lit-c.md, not re-verified)
  gives full fine-tuning default 100 epochs, lr 2e-4, and 5 runs x 500 epochs for the UCR comparison.
* Pretraining corpora (paper, via notes lit-c.md; the arXiv abstract page only says "over a million sequences"):
  1.89M univariate series merged from public data including the UCR (128) and UEA (30, split into univariate) TRAIN
  splits (test splits excluded); their `train_ssl.json` names the dataset `ucr-merged`. UCR/UEA data ARE included.
* Params: see test_nutime.py (measured; the paper/README give no parameter count).
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib
import io
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

COMMIT = "9acdd3d0e82df0914217ef752fa3b95ec82cd960"
CKPT_SHA256 = "65543d55c6a9d30837b46b3809b7358a6ecb3e9702f4eaf01bfd68a96707f7a2"
CKPT_REL = "ckpt/checkpoint_bias9.pth"
_D = 128  # window_emb_dim
_LEN = 512  # transform_size of train_ssl.json / train_ft.json
_WINDOW = 16


def clone_dir():
    from tor.config import get_settings

    return get_settings().cache_dir / "nutime" / "NuTime"


@functools.lru_cache(maxsize=None)
def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _check_clone() -> tuple[bool, str]:
    d = clone_dir()
    if not (d / "src" / "models" / "networks.py").exists():
        return False, f"NuTime clone missing at {d} (git clone github.com/chenguolin/NuTime, checkout {COMMIT})"
    try:
        head = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        return False, f"cannot read NuTime clone commit: {e}"
    if head != COMMIT:
        return False, f"NuTime clone at {head}, expected pinned {COMMIT}"
    if not (d / CKPT_REL).exists():
        return False, f"NuTime checkpoint missing: {d / CKPT_REL}"
    if _sha256(d / CKPT_REL) != CKPT_SHA256:  # hashed once per process (90 MB)
        return False, f"NuTime checkpoint sha256 mismatch: {d / CKPT_REL}"
    return True, ""


def _install_timm_stub() -> None:
    try:
        from timm.models.layers import DropPath, trunc_normal_  # noqa: F401
        return
    except ImportError:
        pass

    class DropPath(nn.Module):
        def __init__(self, *a, **k):
            raise NotImplementedError("timm DropPath stub: NuTime path_dropout must be 0")

    for name in ("timm", "timm.models", "timm.models.layers"):
        m = types.ModuleType(name)
        m.__path__ = []
        m._tor_stub = True
        sys.modules[name] = m
    sys.modules["timm.models.layers"].trunc_normal_ = nn.init.trunc_normal_
    sys.modules["timm.models.layers"].DropPath = DropPath


_THEIRS: SimpleNamespace | None = None


def _their_modules() -> SimpleNamespace:
    """Import WinT, WindowNormEncoder and the transforms of the pinned clone (see module docstring)."""
    global _THEIRS
    ok, why = _check_clone()
    if not ok:
        raise AdapterUnavailable(why)
    if _THEIRS is not None:
        return _THEIRS
    root = clone_dir() / "src"
    is_src = lambda n: n == "src" or n.startswith("src.")  # noqa: E731
    saved = {n: m for n, m in sys.modules.items() if is_src(n)}
    for n in saved:
        del sys.modules[n]
    preexisting_timm = {n for n in sys.modules if n == "timm" or n.startswith("timm.")}
    try:
        _install_timm_stub()
        for name, sub in (("src", ""), ("src.models", "models"), ("src.models.encoders", "models/encoders"),
                          ("src.data", "data")):
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(root / sub) if sub else str(root)]
            sys.modules[name] = pkg
        try:
            networks = importlib.import_module("src.models.networks")
            wne = importlib.import_module("src.models.encoders.WindowNormEncoder")
            transforms = importlib.import_module("src.data.transforms")
        except ImportError as e:
            raise AdapterUnavailable(f"NuTime imports failed: {e}") from e
        _THEIRS = SimpleNamespace(WinT=networks.WinT, WindowNormEncoder=wne.WindowNormEncoder, transforms=transforms)
        return _THEIRS
    finally:
        for n in [n for n in sys.modules if is_src(n)]:
            del sys.modules[n]
        sys.modules.update(saved)
        for n in [n for n, m in list(sys.modules.items()) if getattr(m, "_tor_stub", False)]:
            if n not in preexisting_timm:
                del sys.modules[n]


def _their_transforms():
    return _their_modules().transforms


def _config(series_size: int) -> SimpleNamespace:
    """Config attributes read by WindowNormEncoder / MultiBiasEncoder / WinT, = configs/demo_ft_epilepsy.json
    (+ Config defaults wne_use_in / wne_use_instance_stats False, src/config.py:38-39), univariate (num_channels 1)."""
    return SimpleNamespace(
        model_series_size=series_size, num_channels=1, window_size=_WINDOW, stride=_WINDOW, window_emb_dim=_D,
        wne_use_in=False, wne_use_instance_stats=False, use_ms_embed=True, mean_std_encode="mbe",
        use_channel_emb_sum=False, mb_emb_dim=32, scale_weighted_sum=True, loc_weighted_sum=False,
        use_loc_emb=False, num_bias=9, base_bias=10, multibias_trainable=True, scales=None, locs=None,
        transformer_mask_type="none", transformer_mask_scale=0.2, encoder="wne",
    )


def build_their_model(series_size: int, n_out: int = 2) -> nn.Sequential:
    """`nn.Sequential(WindowNormEncoder, WinT)` as in ClassificationExp.build_model (random init, fine-tuning config:
    depth 6, heads 8, head_dim 64, mlp 512, fixed PE, cls pooling, dropout 0.1, attn_dropout 0.1)."""
    t = _their_modules()
    cfg = _config(series_size)
    with contextlib.redirect_stdout(io.StringIO()):  # MultiBiasEncoder prints its scales/locs
        encoder = t.WindowNormEncoder(cfg, None, None)
    seq_len = encoder.window_num
    wint = t.WinT(
        cfg, series_size=seq_len, in_channels=_D, window_emb_dim=_D, out_channels=n_out, window_size=_WINDOW,
        stride=_WINDOW, depth=6, heads=8, head_dim=64, mlp_dim=512, pool_mode="cls", pe_mode="fixed",
        max_num_tokens=1024, dropout=0.1, pos_dropout=0.0, attn_dropout=0.1, path_dropout=0.0, init_values=None,
    )
    return nn.Sequential(encoder, wint)


def checkpoint_state_dict() -> dict:
    """Checkpoint state remapped like `cls_exp.py::load_pretrained_model`: strip `backbone.`; the BYOL projector
    (`backbone.1.fc.*`), momentum copy and predictor are not part of the classification model."""
    path = clone_dir() / CKPT_REL
    if _sha256(path) != CKPT_SHA256:
        raise AdapterUnavailable(f"NuTime checkpoint sha256 mismatch: {path}")
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]  # hash-verified pickle
    return {k[len("backbone."):]: v for k, v in sd.items()
            if k.startswith("backbone.") and not k.startswith("backbone.1.fc.")}


def load_pretrained_into(model: nn.Module, sd: dict) -> None:
    """strict load except for the fresh classification head `1.fc.*`."""
    res = model.load_state_dict(sd, strict=False)
    bad = [k for k in res.missing_keys if not k.startswith("1.fc.")]
    if bad or res.unexpected_keys:
        raise RuntimeError(f"NuTime checkpoint mismatch: missing {bad}, unexpected {res.unexpected_keys}")


def _wrap_checkpoint(block: nn.Module) -> None:
    orig = block.forward

    def fwd(*a, **k):
        if block.training and torch.is_grad_enabled():
            return checkpoint(orig, *a, use_reentrant=False, **k)
        return orig(*a, **k)

    block.forward = fwd


class _Backbone(nn.Module):
    """(n, C, L) -> (n, C*128): their `WinT.features` (LayerNorm'd cls token) per channel."""

    def __init__(self, full: nn.Sequential):
        super().__init__()
        self.encoder, self.wint = full[0], full[1]
        self.wint.fc = nn.Identity()  # their fc is not part of the embedding; WinT.forward now returns `features`

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, length = x.shape
        out = self.wint(self.encoder(x.reshape(n * c, 1, length)))
        return out.reshape(n, c * _D)


class _Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, n_classes: int, n_channels: int, head_mode: bool):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(_D * n_channels, n_classes)  # their `fc`
        if head_mode:  # cls_exp.py:63-68 (freeze_backbone branch)
            nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


@register
class NuTime(FMAdapter):
    name = "nutime"
    hf_repo = None  # GitHub-hosted checkpoint (no HF repo)
    max_length = _LEN
    multivariate = "per_channel"
    supported_modes: tuple[Mode, ...] = ("head", "lora", "full", "scratch")
    supports_grad_ckpt = True
    lora_targets = ("to_qkv", "to_out.0")
    pretraining_corpora = (
        "NuTime paper (Lin et al., arXiv 2310.07402, TMLR 2024), via notes lit-c.md (arXiv abstract: 'over a million "
        "sequences'): 1.89M univariate series merged from public data (repo config dataset 'ucr-merged'), including "
        "the TRAIN splits of the UCR (128) and UEA (30, split into univariate) archives; UCR/UEA data ARE included "
        "(test splits excluded; overlap with downstream UCR train sets not discussed)."
    )
    recipe_source = (
        "configs/demo_ft_epilepsy.json of github.com/chenguolin/NuTime @ " + COMMIT[:10] + ": AdamW lr 2e-4, "
        "weight_decay 0.05, num_epochs 100, batch_size 2048, cosine with 10 warmup epochs (dataset-specific demo for "
        "Epilepsy; train_ft.json differs: lr 2e-3, bs 1024, 200 epochs; paper full-FT default 100 ep, lr 2e-4)"
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
            _their_modules()
        except AdapterUnavailable as e:
            return False, str(e)
        return True, ""

    @staticmethod
    def _build(seed: int, length: int, n_out: int = 2) -> nn.Sequential:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return build_their_model(length, n_out)

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
        self._model = self._their_model(_LEN)  # reference model (params are independent of series length)

    def _their_model(self, length: int, seed: int | None = None) -> nn.Sequential:
        """Their `nn.Sequential(encoder, WinT)` (2-class fc) with the loaded init, for any length % 16 == 0."""
        m = self._build(self._seed if seed is None else seed, length)
        if self._init_kind == "pretrained":
            load_pretrained_into(m, self._sd)
        return m

    def unused_param_count(self) -> int:
        """Params of the loaded their-model (2-class `fc`) that the classifier's embedding path does not use."""
        if self._init_kind is None:
            raise RuntimeError("call load() first")
        return sum(p.numel() for n, p in self._model.named_parameters() if n.startswith("1.fc."))

    def preprocess(self, X: np.ndarray) -> torch.Tensor:
        x = np.asarray(X, dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError(f"{self.name}: input contains NaN/inf; impute first")
        if x.ndim == 2:
            x = x[:, None, :]
        t = torch.as_tensor(np.ascontiguousarray(x))
        # TSResize(size=512, mode='linear'): F.interpolate, align_corners=False (transforms.py:179)
        return F.interpolate(t, size=_LEN, mode="linear", align_corners=False)

    def embed(self, X: torch.Tensor) -> torch.Tensor:
        if self._init_kind is None:
            raise RuntimeError("call load() first")
        length = X.shape[-1]
        if length not in self._cache:
            self._cache[length] = _Backbone(self._their_model(length)).eval()
        with torch.no_grad():
            return self._cache[length](X)

    def classifier(self, n_classes, mode, n_channels, seq_len, grad_ckpt=False, seed=0):
        self._check_mode(mode, grad_ckpt)
        if mode != "scratch" and self._init_kind != "pretrained":
            raise RuntimeError(f"{mode} needs load('pretrained') first")
        full = self._build(seed, _LEN)  # seq_len is ignored: preprocess always resizes to 512
        if mode != "scratch":
            load_pretrained_into(full, self._sd)
        backbone = _Backbone(full)
        for p in backbone.parameters():
            p.requires_grad = True
        if grad_ckpt:
            for blk in backbone.wint.transformer.layers:
                if hasattr(blk, "fn"):  # PreNorm attention / feed-forward blocks (others are Identity)
                    _wrap_checkpoint(blk)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if mode == "lora":
                import peft

                backbone = peft.get_peft_model(backbone, self.lora_config())
            elif mode == "head":
                for p in backbone.parameters():
                    p.requires_grad = False
            return _Classifier(backbone, n_classes, n_channels, mode == "head")

    def default_recipe(self) -> Recipe:
        return Recipe(lr=2e-4, weight_decay=0.05, epochs=100, batch_size=2048, warmup_frac=0.1)
