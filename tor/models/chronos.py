"""Chronos adapters: `chronos-bolt-small`, `chronos-bolt-base` (amazon/chronos-bolt-{small,base}) and
`chronos2` (amazon/chronos-2). Package `chronos-forecasting` 2.3.2 (repo amazon-science/chronos-forecasting);
its deps (torch<3, transformers<6, ...) are compatible with the env, so it was installed normally.

Chronos-Bolt (`chronos.chronos_bolt.ChronosBoltModelForForecasting`): T5 encoder-decoder forecaster.
Classification uses the ENCODER only: `model.encode(context)` (instance standardisation -> patching
(16, stride 16) -> `input_patch_embedding` -> [REG] token appended -> T5 encoder), tokens mean-pooled,
as Jain et al. (arXiv 2611.06988, "Frozen Forecasters as Classifiers on MONSTER") use frozen
Chronos-Bolt features (title/authors/abstract checked; the encoder-only + mean-pooling detail was NOT
verified against the paper text, it is the task specification). The decoder and
`output_patch_embedding` are deleted from classifiers (unused, so trainable counts are honest).
Univariate: per channel, embeddings concatenated (d = d_model * C).

Chronos-2 (`chronos.chronos2.Chronos2Model`): encoder-only (alternating time attention / group
attention / FFN blocks). Natively multivariate: `Chronos2Pipeline.embed` (chronos2/pipeline.py)
feeds each variate as a row of `Chronos2Model.encode(context=(n*C, L), group_ids=...)` where all rows of
one series share a group id, so the group-attention layers mix the channels of a series; it returns
(n_variates, num_patches + 2, d). The adapter does exactly this and mean-pools tokens per row; the C
row embeddings are concatenated (d = 768 * C, head sized from n_channels). Pooled tokens: the context
patches plus the [REG] token; the final masked future-output patch token (placeholder for the forecast
horizon, `encode` always appends one) is excluded. Chronos-2 internal scaling: standardise then arcsinh.

Length (`preprocess`): both packages truncate to the last `context_length` observations
(chronos_bolt.py `encode`: `context[..., -context_length:]`, 2048; chronos2/model.py
`_prepare_patched_context`: 8192) and left-pad with NaN up to a multiple of the patch size inside `Patch`
(NaN = padding/masking in these models). `preprocess` therefore only truncates to the last
`max_length` points (no resizing) and the models pad themselves; it REJECTS NaN input because NaN is
the models' missing-value marker (it would be silently masked). Shorter series need no minimum.

Scratch: the same architecture from the checkpoint's HF config (`AutoConfig` + `Model(config)`; the
package's own `_init_weights`), seeded torch RNG; no pretrained weights.

LoRA: peft on attention projections `q`,`k`,`v`,`o`. Bolt: encoder.block.N.layer.0.SelfAttention.{q,k,v,o}.
Chronos-2: encoder.block.N.layer.{0,1}.self_attention.{q,k,v,o} (time AND group attention; the package's
own `Chronos2Pipeline.fit` LoRA default also targets `self_attention.q/k/v/o`, plus the output head,
which is not present here). Head trainable.

Gradient checkpointing: torch.utils.checkpoint (non-reentrant) around each encoder block's forward,
active in train mode only (neither package implements checkpointing itself).

Params (loaded pretrained, with decoder/forecast head): small 47,718,016; base 205,292,928; chronos-2
119,477,664 (measured; match the model cards' 48M / 205M / 120M). Classifier counts exclude decoder
and forecast head.

Pretraining corpora: see class attributes. Recipe: no published classification fine-tuning recipe was
found (verified only for the package: `Chronos2Pipeline.fit` is a FORECASTING fine-tune, lr 1e-6,
1000 steps; no classification code) -> harness default.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tor.models import register
from tor.models.base import AdapterUnavailable, FMAdapter, Mode, Recipe

try:
    from chronos.chronos2 import Chronos2Model
    from chronos.chronos_bolt import ChronosBoltModelForForecasting
except ImportError as e:  # registry skips the module
    raise AdapterUnavailable(f"chronos-forecasting not installed: {e}") from e


def _hub_kwargs() -> dict:
    from tor.config import get_settings, load_secrets

    load_secrets()
    cache = get_settings().cache_dir / "hf"
    cache.mkdir(parents=True, exist_ok=True)
    return {"cache_dir": str(cache)}


def _wrap_checkpoint(block: nn.Module) -> None:
    orig = block.forward

    def fwd(*a, **k):
        if block.training and torch.is_grad_enabled():
            return checkpoint(orig, *a, use_reentrant=False, **k)
        return orig(*a, **k)

    block.forward = fwd


class _Backbone(nn.Module):
    """(n, C, L) -> (n, C*d): per-row encoder tokens mean-pooled, channels concatenated."""

    def __init__(self, model: nn.Module, kind: str):
        super().__init__()
        self.model = model
        self.kind = kind
        self.d_model = model.config.d_model
        self.patch = model.chronos_config.input_patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, length = x.shape
        rows = x.reshape(n * c, length)
        if self.kind == "bolt":
            hidden = self.model.encode(rows)[0]  # (n*c, patches [+REG], d); all tokens real
            pooled = hidden.mean(dim=1)
        else:
            gid = torch.arange(n, device=x.device).repeat_interleave(c)
            enc, _, _, n_ctx = self.model.encode(context=rows, group_ids=gid)
            hidden = enc[0]  # (n*c, n_ctx [+REG] + 1 future-output patch, d)
            n_keep = hidden.shape[1] - 1  # drop the masked future-output placeholder token
            pooled = hidden[:, :n_keep].mean(dim=1)
        return pooled.reshape(n, c * self.d_model)


class _Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, d_model: int, n_classes: int, n_channels: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(d_model * n_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class _ChronosBase(FMAdapter):
    hf_repo: str
    kind: str  # "bolt" | "chronos2"
    supported_modes: tuple[Mode, ...] = ("head", "lora", "full", "scratch")
    supports_grad_ckpt = True
    lora_targets = ("q", "k", "v", "o")
    recipe_source = (
        "harness default: no published classification fine-tuning recipe for Chronos-Bolt / Chronos-2 "
        "(Ansari et al. arXiv 2403.07815 and 2510.15821 evaluate forecasting only; the package's "
        "Chronos2Pipeline.fit is a forecasting fine-tune with lr 1e-6, 1000 steps, batch 256, not "
        "classification). Search of the papers/repo only; not an exhaustive literature search"
    )

    def __init__(self):
        self._model = None
        self._init_kind = None
        self._config = None

    def _cls(self):
        return Chronos2Model if self.kind == "chronos2" else ChronosBoltModelForForecasting

    def _random_model(self, seed: int):
        from transformers import AutoConfig

        if self._config is None:
            self._config = AutoConfig.from_pretrained(self.hf_repo, **_hub_kwargs())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return self._cls()(copy.deepcopy(self._config))

    def load(self, init) -> None:
        if init == "pretrained":
            m = self._cls().from_pretrained(self.hf_repo, **_hub_kwargs())
            self._init_kind = "pretrained"
        else:
            kind, seed = init
            if kind != "random":
                raise ValueError(f"bad init {init!r}")
            m = self._random_model(seed)
            self._init_kind = "random"
        self._model = m.eval().float()
        self._backbone = _Backbone(self._model, self.kind).eval()

    def unused_param_count(self) -> int:
        """Params of decoder / forecast head removed from classifiers."""
        if self._model is None:
            raise RuntimeError("call load() first")
        m = self._model
        mods = [m.output_patch_embedding] + ([m.decoder] if self.kind == "bolt" else [])
        seen = {id(p) for p in m.encoder.parameters()} | {id(p) for p in m.shared.parameters()}
        seen |= {id(p) for p in m.input_patch_embedding.parameters()}
        return sum(p.numel() for mod in mods for p in mod.parameters() if id(p) not in seen)

    def preprocess(self, X: np.ndarray) -> torch.Tensor:
        x = np.asarray(X, dtype=np.float32)
        if np.isnan(x).any():
            raise ValueError(f"{self.name}: input contains NaN, which the model treats as padding; impute first")
        if x.ndim == 2:
            x = x[:, None, :]
        if x.shape[-1] > self.max_length:  # models keep the last context_length points
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
            model = copy.deepcopy(self._model)
        model.output_patch_embedding = None  # forecast head, unused
        if self.kind == "bolt":
            model.decoder = None  # unused by encode()
        for p in model.parameters():
            p.requires_grad = True
        if grad_ckpt:
            for blk in model.encoder.block:
                _wrap_checkpoint(blk)
        backbone = _Backbone(model, self.kind)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if mode == "lora":
                import peft

                backbone = peft.get_peft_model(backbone, self.lora_config())
            elif mode == "head":
                for p in backbone.parameters():
                    p.requires_grad = False
            return _Classifier(backbone, model.config.d_model, n_classes, n_channels)

    def default_recipe(self) -> Recipe:
        return Recipe(lr=1e-4, weight_decay=0.01, epochs=100, batch_size=32)


class _Bolt(_ChronosBase):
    kind = "bolt"
    max_length = 2048
    multivariate = "per_channel"
    pretraining_corpora = (
        "Chronos-Bolt model card (huggingface.co/amazon/chronos-bolt-small): trained on 'nearly 100 "
        "billion time series observations'. The card does not list the corpora; the Chronos paper "
        "(Ansari et al., arXiv 2403.07815) documents the original Chronos corpus (public + KernelSynth "
        "synthetic), and that Bolt reuses it is inferred, not verified. UCR/UEA inclusion: not documented."
    )


class _Chronos2(_ChronosBase):
    kind = "chronos2"
    max_length = 8192
    multivariate = "native"
    pretraining_corpora = (
        "Chronos-2 model card (huggingface.co/amazon/chronos-2, 'Training data') and Ansari et al., "
        "arXiv 2510.15821: subset of autogluon/chronos_datasets (excluding test portions overlapping "
        "GIFT-Eval), subset of Salesforce/GiftEvalPretrain, and synthetic univariate and multivariate "
        "data. UCR/UEA classification archives are not listed on the card; whether any are inside "
        "those subsets: not documented (not verified)."
    )


@register
class ChronosBoltSmall(_Bolt):
    name = "chronos-bolt-small"
    hf_repo = "amazon/chronos-bolt-small"


@register
class ChronosBoltBase(_Bolt):
    name = "chronos-bolt-base"
    hf_repo = "amazon/chronos-bolt-base"


@register
class Chronos2(_Chronos2):
    name = "chronos2"
    hf_repo = "amazon/chronos-2"
