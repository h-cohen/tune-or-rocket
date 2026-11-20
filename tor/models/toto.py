"""Toto adapter: `toto` (Datadog/Toto-Open-Base-1.0).

Package: `toto-ts` 0.2.0 from PyPI (repo DataDog/toto; paper "This Time is Different: An Observability
Perspective on Time Series Foundation Models", Cohen et al., arXiv 2505.14766, verified title/ID). Installed
with `--no-deps`: its pins (torch==2.7.0, transformers==4.52.1, numpy==1.26.4, ...) would downgrade the env.
Imports additionally need `gluonts` 0.17.0 (scaler base class + distributions), `lightning` 2.6.6 (gluonts.torch
imports it), `jaxtyping`, `rotary-embedding-torch`, all installed without touching torch/transformers/peft.
xFormers is not installed, so the model is built with `use_memory_efficient_attention=False` (torch SDPA path
inside the package's own attention code) and the unfused SwiGLU MLP (`mlp.0`, `mlp.2`).

Loading: `Toto.load_from_checkpoint(<snapshot dir with config.json + model.safetensors>)` (strict load of the
checkpoint keys; the package's `from_pretrained` is incompatible with the installed huggingface_hub 2.x:
`_from_pretrained()` misses the `proxies`/`resume_download` kwargs).

Architecture (config.json): patch size/stride 64, embed_dim 768, 12 layers, 12 heads, MLP 3072 (SwiGLU),
dropout 0.1, `CausalPatchStdMeanScaler`, decoder-only. Layers 0-10 are time-wise (causal, rotary) attention,
layer 11 is space-wise (bidirectional across variates at each patch position, no rotary): `spacewise_every_n_layers`
=12, `spacewise_first`=False. Each attention is `attention.wQKV` (fused q,k,v) + `attention.wO`. After the
transformer: `unembed` (768 -> 768*64) and a Student-T mixture head (`output_distribution`, 24 components).

Embedding (`_Backbone`): replicates `TotoBackbone.backbone` (backbone.py) up to the transformer output: the
package's own scaler (`self.scaler`, the causal patch-wise StdMean scaler, called with padding_mask and unit
weights exactly as in `backbone`) -> `patch_embed` -> `transformer`. The transformer output (768-d per patch,
BEFORE `unembed` and the output distribution) is mean-pooled over the real patch tokens of each variate
(patches that are not entirely padding). Natively multivariate: the C channels are fed as the C variates of
one sample (`id_mask` all ones = one group), so the space-wise layer lets channels attend to each other.
RULING: pooling is over time patches per variate, then variates are concatenated, d = 768 * C (head sized
from n_channels). Mean pooling is a HARNESS choice: the package/paper define no classification head (the paper
uses embeddings only in an analysis of the model, not verified as a recipe) and with causal time attention
early tokens see little context, so last-token pooling is plausible and was not evaluated.

Length (`preprocess` + `_Backbone.pad`): the package (`pad_array` / `pad_id_mask` in toto/data/util/dataset.py,
used by `TotoForecaster.forecast`) LEFT-pads with zeros to a multiple of the patch stride 64 with
padding_mask False and id_mask repeating the left edge; we do the same (<64 points, so never a fully padded
patch; the exclusion in pooling only matters for package-style longer padding). Note: as in the package,
padded patches are not masked out of time attention (only the scaler uses padding_mask). The code has NO
hard context limit (rotary, no learned positions); `max_length` = 4096 is the context length the paper
(Sec. on embeddings: "context length L=4096") and README example (`torch.randn(7, 4096)`) use; whether
4096 was the pretraining context is not verified. Longer series keep the LAST 4096 points (the package
forecasts from the most recent context). `preprocess` REJECTS NaN/inf (the package takes a padding_mask for
missing data but we do not use it for NaN; impute first).

Scratch: `Toto(**config.json)` constructed under a seeded torch RNG: PyTorch default init (nn.Linear
Kaiming-uniform, RMSNorm scale 1); the package defines no custom initialiser. Config comes from the HF
config.json (weights are not downloaded for scratch).

LoRA: peft on `wQKV` and `wO` (names shared by the 11 time-wise and the 1 space-wise attention layers):
`model.transformer.layers.N.attention.{wQKV,wO}`, N in 0..11. Head trainable.

Gradient checkpointing: torch.utils.checkpoint (non-reentrant) around each transformer layer, training only
(the package implements none).

Params (measured): pretrained total 151,306,080 (paper abstract: "151 million parameters") including
`unembed` and `output_distribution`; both are deleted from classifiers (unused by the embedding);
`unused_param_count()` reports their size, so full = total - unused + head.

Recipe: the repo ships a FORECASTING fine-tuning pipeline (toto/scripts/finetune_toto.py, lr 1e-4 in its
config) but no classification recipe -> harness default.
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tor.models import register
from tor.models.base import AdapterUnavailable, FMAdapter, Mode, Recipe

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from toto.model.toto import Toto as _Toto
except ImportError as e:  # registry skips the module
    raise AdapterUnavailable(f"toto-ts (or gluonts/lightning/jaxtyping/rotary-embedding-torch) not installed: {e}") from e

_REPO = "Datadog/Toto-Open-Base-1.0"
_PATCH = 64
_MAX_LENGTH = 4096


def _snapshot_dir(weights: bool = True) -> str:
    """Directory holding config.json (and model.safetensors if `weights`) in the TOR_CACHE HF cache."""
    from huggingface_hub import hf_hub_download

    from tor.config import get_settings, load_secrets

    load_secrets()
    cache = get_settings().cache_dir / "hf"
    cache.mkdir(parents=True, exist_ok=True)
    cfg = hf_hub_download(_REPO, "config.json", cache_dir=str(cache))
    if weights:
        hf_hub_download(_REPO, "model.safetensors", cache_dir=str(cache))
    return os.path.dirname(cfg)


def _config() -> dict:
    with open(os.path.join(_snapshot_dir(weights=False), "config.json")) as f:
        cfg = json.load(f)
    cfg["use_memory_efficient_attention"] = False  # xFormers not installed
    return cfg


def _wrap_checkpoint(block: nn.Module) -> None:
    orig = block.forward

    def fwd(*a, **k):
        if block.training and torch.is_grad_enabled():
            return checkpoint(orig, *a, use_reentrant=False, **k)
        return orig(*a, **k)

    block.forward = fwd


class _Backbone(nn.Module):
    """(n, C, L) -> (n, C*768): transformer output, mean over real patches per variate, variates concatenated."""

    def __init__(self, toto: nn.Module):
        super().__init__()
        self.toto = toto  # `Toto` wrapper; `toto.model` is the TotoBackbone
        self.d_model = toto.model.embed_dim

    @staticmethod
    def pad(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Left-pad (n, C, L) with zeros to a multiple of 64 (`pad_array`); padding_mask True = real."""
        length = x.shape[-1]
        front = -length % _PATCH
        mask = torch.ones_like(x, dtype=torch.bool)
        if front:
            x = torch.cat([x.new_zeros(*x.shape[:-1], front), x], dim=-1)
            mask = torch.cat([torch.zeros(*mask.shape[:-1], front, dtype=torch.bool, device=x.device), mask], dim=-1)
        return x, mask

    def tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transformer output (n, C, P, 768) and (n, C, P) bool of real (not fully padded) patches."""
        bb = self.toto.model
        series, real = self.pad(x)
        id_mask = torch.ones_like(series)  # one group; pad_id_mask repeats the left edge = ones
        scaled, _, _ = bb.scaler(series, weights=torch.ones_like(series), padding_mask=real, prefix_length=None)
        emb, reduced_id_mask = bb.patch_embed(scaled, id_mask)
        hidden = bb.transformer(emb, reduced_id_mask, None, variate_label_embeds=None)
        valid = real.unfold(-1, _PATCH, _PATCH).any(-1)
        return hidden, valid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, valid = self.tokens(x)
        w = valid.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * w).sum(dim=2) / w.sum(dim=2).clamp(min=1.0)  # (n, C, 768)
        return pooled.flatten(1)


class _Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, d_model: int, n_classes: int, n_channels: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(d_model * n_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def _strip_unused(toto: nn.Module) -> None:
    toto.model.unembed = None  # forecast head path, unused by the embedding
    toto.model.output_distribution = None


@register
class TotoAdapter(FMAdapter):
    name = "toto"
    hf_repo = _REPO
    max_length = _MAX_LENGTH
    multivariate = "native"
    supported_modes: tuple[Mode, ...] = ("head", "lora", "full", "scratch")
    supports_grad_ckpt = True
    lora_targets = ("wQKV", "wO")
    pretraining_corpora = (
        "Toto paper (Cohen et al., arXiv 2505.14766): about 2.36 trillion time series points (1.59 trillion "
        "non-repeated, non-synthetic); 43% anonymous Datadog observability metrics; public data (GIFT-Eval "
        "Pretrain and the Chronos collections minus the subset overlapping the GIFT-Eval benchmark); plus "
        "synthetic data (proportion not read). UCR/UEA classification archives: not documented (not mentioned "
        "in the parts of the paper read; the last ~48% of the HTML text and the appendices were not read)."
    )
    recipe_source = (
        "harness default: the DataDog/toto repo ships a forecasting fine-tuning pipeline only "
        "(toto/scripts/finetune_toto.py); no classification fine-tuning recipe found in the paper/README/repo "
        "(checked, not exhaustive)"
    )

    def __init__(self):
        self._model = None
        self._init_kind = None
        self._backbone = None

    @staticmethod
    def _random_model(seed: int) -> nn.Module:
        cfg = _config()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return _Toto(**cfg)

    def load(self, init) -> None:
        if init == "pretrained":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = _Toto.load_from_checkpoint(_snapshot_dir(), map_location="cpu", strict=True,
                                               use_memory_efficient_attention=False)
            self._init_kind = "pretrained"
        else:
            kind, seed = init
            if kind != "random":
                raise ValueError(f"bad init {init!r}")
            m = self._random_model(seed)
            self._init_kind = "random"
        self._model = m.eval().float()
        self._backbone = _Backbone(self._model).eval()

    def unused_param_count(self) -> int:
        """Params of `unembed` and the Student-T mixture head, removed from classifiers."""
        if self._model is None:
            raise RuntimeError("call load() first")
        bb = self._model.model
        return sum(p.numel() for mod in (bb.unembed, bb.output_distribution) for p in mod.parameters())

    def preprocess(self, X: np.ndarray) -> torch.Tensor:
        x = np.asarray(X, dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError(f"{self.name}: input contains NaN/inf; impute first")
        if x.ndim == 2:
            x = x[:, None, :]
        if x.shape[-1] > self.max_length:  # keep the most recent context
            x = x[..., -self.max_length:]
        return torch.as_tensor(np.ascontiguousarray(x))

    def embed(self, X: torch.Tensor) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("call load() first")
        self._backbone.eval()
        with torch.no_grad():
            return self._backbone(X)

    def classifier(self, n_classes, mode, n_channels, seq_len, grad_ckpt=False, seed=0):
        self._check_mode(mode, grad_ckpt)
        if mode == "scratch":
            model = self._random_model(seed)
        else:
            if self._init_kind != "pretrained":
                raise RuntimeError(f"{mode} needs load('pretrained') first")
            model = self._random_model(seed)  # deepcopy fails (gluonts scaler): rebuild + copy weights
            model.load_state_dict(self._model.state_dict(), strict=True)
        _strip_unused(model)
        for p in model.parameters():
            p.requires_grad = True
        if grad_ckpt:
            for blk in model.model.transformer.layers:
                _wrap_checkpoint(blk)
        backbone = _Backbone(model)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if mode == "lora":
                import peft

                backbone = peft.get_peft_model(backbone, self.lora_config())
            elif mode == "head":
                for p in backbone.parameters():
                    p.requires_grad = False
            return _Classifier(backbone, model.model.embed_dim, n_classes, n_channels)

    def default_recipe(self) -> Recipe:
        return Recipe(lr=1e-4, weight_decay=0.01, epochs=100, batch_size=32)

