"""TimesFM-2.5 adapter: `timesfm25` (google/timesfm-2.5-200m-pytorch).

Package: `timesfm` 3.0.2 from PyPI (repo google-research/timesfm; the wheel contains both the TimesFM-3.0
code and the 2.5 code in `timesfm/timesfm_2p5/`). Installed with `--no-deps`: its hard deps (numpy,
huggingface_hub, safetensors) were already present and the `torch` extra only requires torch>=2.0.
The adapter uses `timesfm.timesfm_2p5.timesfm_2p5_torch.TimesFM_2p5_200M_torch_module` (the raw nn.Module) and
the package's `timesfm.torch.util` helpers. It does not use the `TimesFM_2p5_200M_torch` wrapper because that
wrapper's `load_checkpoint` moves the model to CUDA when available and applies torch.compile.

Architecture: decoder-only (causal attention) patched transformer: input patch length 32; `tokenizer`
(ResidualBlock on [patch values, patch mask], 64 -> 1280); 20 `stacked_xf` Transformer layers (model dim 1280,
16 heads, fused `attn.qkv_proj`, `attn.out`, RMS norms, rotary positions); then `output_projection_point`
and `output_projection_quantiles` (forecast heads; output patch 128, quantile len 1024).

Embedding (`_Backbone`): replicates the prefill step of the package's own `module.decode`
(timesfm_2p5_torch.py): per-patch running mean/std (`util.update_running_stats`, masks respected) ->
`util.revin` instance normalisation -> masked positions zeroed -> `tokenizer` -> `stacked_xf`. The output
of the last stacked transformer layer (the second element, `output_embeddings`, of `module.forward`'s
embedding tuple; before the output projections) is mean-pooled over the real patch tokens (patches that
are not entirely padding). Mean pooling is a HARNESS choice: the package/model card define no
classification or embedding path, and with causal attention the early tokens see little context, so other
pooling (e.g. last token) is plausible and was not evaluated. Univariate: per channel, embeddings
concatenated (d = 1280 * C). The two forecast-head ResidualBlocks are deleted from classifiers (unused by
the embedding, so trainable counts are honest); `unused_param_count()` reports their size.

Length (`preprocess` + `_Backbone.pad`): `TimesFM_2p5.forecast` (timesfm_2p5_base.py) keeps the last
`max_context` points of longer series and LEFT-pads shorter ones with zeros plus a True (=masked) mask to
the fixed context length (`forecast_naive` in timesfm_2p5_torch.py pads only to a multiple of the patch
size 32). We truncate to the last `max_length` = 16384 (`TimesFM_2p5_200M_Definition.context_limit`; the
package requires context + horizon <= 16384 in `compile`, here no horizon is decoded) and pad to a
multiple of 32 like `forecast_naive`; pad positions are masked (excluded from attention keys and from running
stats). The model treats mask, not NaN, as padding, but `forecast` interpolates NaNs
(`linear_interpolation`), so `preprocess` REJECTS NaN (impute first) and no NaN is ever passed.

Scratch: same architecture, `TimesFM_2p5_200M_torch_module()` constructed under a seeded torch RNG. The
package defines no custom initialiser, so this is PyTorch default init (nn.Linear Kaiming-uniform, RMSNorm /
PerDimScale package defaults), no pretrained weights.

LoRA: peft on `qkv_proj` (fused q,k,v projection; the model has no separate q/k/v modules) and `out`
(attention output projection): `stacked_xf.N.attn.{qkv_proj,out}`, N in 0..19. Head trainable.

Gradient checkpointing: torch.utils.checkpoint (non-reentrant) around each `stacked_xf` layer, training
mode only (the package implements none).

Params (measured): loaded pretrained total 231,289,280 (with forecast heads) - unused forecast heads
32,768,000 = 198,521,280 used by the classifier backbone, which matches the "200M" in the model name by
measured subtraction (the model card does not itemise).

Note: `_Backbone.pad` pads fewer than 32 points, so it never yields a fully padded patch; the
fully-padded-patch exclusion in pooling only matters for package-style masks (padding to a fixed context).

Recipe: no published classification fine-tuning recipe was found (the TimesFM paper, arXiv 2310.10688, and
the repo evaluate forecasting; search of paper/repo/model card only, not exhaustive) -> harness default.
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
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module as _Module
    from timesfm.torch import util as _util
except ImportError as e:  # registry skips the module
    raise AdapterUnavailable(f"timesfm not installed: {e}") from e

_REPO = "google/timesfm-2.5-200m-pytorch"
_WEIGHTS = "model.safetensors"
_PATCH = 32
_CONTEXT_LIMIT = 16384


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
    """(n, C, L) -> (n, C*1280): last stacked-transformer layer, mean over real patch tokens."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.d_model = model.md

    @staticmethod
    def pad(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Left-pad (R, L) with zeros to a multiple of 32; mask True = padding (as `forecast_naive`)."""
        mask = torch.zeros_like(rows, dtype=torch.bool)
        front = -rows.shape[-1] % _PATCH
        if front:
            rows = torch.cat([rows.new_zeros(rows.shape[0], front), rows], dim=-1)
            mask = torch.cat([torch.ones(rows.shape[0], front, dtype=torch.bool, device=rows.device), mask], dim=-1)
        return rows, mask

    def tokens(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Hidden states (R, P, 1280) and a (R, P) bool of real (not fully padded) patches."""
        m = self.model
        inputs, masks = self.pad(rows)
        r = inputs.shape[0]
        patched = inputs.reshape(r, -1, m.p)
        pmask = masks.reshape(r, -1, m.p)
        n = torch.zeros(r, device=inputs.device)
        mu = torch.zeros(r, device=inputs.device)
        sigma = torch.zeros(r, device=inputs.device)
        mus, sigmas = [], []
        for i in range(patched.shape[1]):  # running stats, as module.decode prefill
            (n, mu, sigma), _ = _util.update_running_stats(n, mu, sigma, patched[:, i], pmask[:, i])
            mus.append(mu)
            sigmas.append(sigma)
        mu_t, sigma_t = torch.stack(mus, dim=1), torch.stack(sigmas, dim=1)
        normed = _util.revin(patched, mu_t, sigma_t, reverse=False)
        normed = torch.where(pmask, 0.0, normed)
        h = m.tokenizer(torch.cat([normed, pmask.to(normed.dtype)], dim=-1))
        for layer in m.stacked_xf:
            h, _ = layer(h, pmask[..., -1], None)
        return h, ~pmask[..., -1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, length = x.shape
        hidden, valid = self.tokens(x.reshape(n * c, length))
        w = valid.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        return pooled.reshape(n, c * self.d_model)


class _Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, d_model: int, n_classes: int, n_channels: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(d_model * n_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


@register
class TimesFM25(FMAdapter):
    name = "timesfm25"
    hf_repo = _REPO
    max_length = _CONTEXT_LIMIT
    multivariate = "per_channel"
    supported_modes: tuple[Mode, ...] = ("head", "lora", "full", "scratch")
    supports_grad_ckpt = True
    lora_targets = ("qkv_proj", "out")
    pretraining_corpora = (
        "TimesFM-2.5 model card (huggingface.co/google/timesfm-2.5-200m-pytorch) and the TimesFM paper "
        "(Das et al., ICML 2024, arXiv 2310.10688, Sec. 'Pretraining Data'): Google Trends, Wikipedia "
        "pageviews, synthetic data, plus public sets (M4, electricity, traffic, weather, Favorita, LibCity, "
        "etc.) for the paper's 200M model. There is no separate 2.5 paper; the exact 2.5 corpus is not "
        "described in the sources checked, so the paper's list is not verified for 2.5. UCR/UEA "
        "classification archives: not documented."
    )
    recipe_source = (
        "harness default: no published classification fine-tuning recipe for TimesFM (arXiv 2310.10688 and "
        "the google-research/timesfm repo cover forecasting only; paper/repo/model card checked, not an "
        "exhaustive literature search)"
    )

    def __init__(self):
        self._model = None
        self._init_kind = None

    @staticmethod
    def _random_model(seed: int) -> nn.Module:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return _Module()

    def load(self, init) -> None:
        if init == "pretrained":
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            path = hf_hub_download(_REPO, _WEIGHTS, **_hub_kwargs())
            m = _Module()
            m.load_state_dict(load_file(path), strict=True)  # (not module.load_checkpoint: it moves to CUDA)
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
        """Params of the two forecast-head ResidualBlocks removed from classifiers."""
        if self._model is None:
            raise RuntimeError("call load() first")
        m = self._model
        return sum(p.numel() for mod in (m.output_projection_point, m.output_projection_quantiles)
                   for p in mod.parameters())

    def preprocess(self, X: np.ndarray) -> torch.Tensor:
        x = np.asarray(X, dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError(f"{self.name}: input contains NaN/inf; impute first")
        if x.ndim == 2:
            x = x[:, None, :]
        if x.shape[-1] > self.max_length:  # package keeps the last max_context points
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
        model.output_projection_point = None  # forecast heads, unused
        model.output_projection_quantiles = None
        for p in model.parameters():
            p.requires_grad = True
        if grad_ckpt:
            for blk in model.stacked_xf:
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
            return _Classifier(backbone, model.md, n_classes, n_channels)

    def default_recipe(self) -> Recipe:
        return Recipe(lr=1e-4, weight_decay=0.01, epochs=100, batch_size=32)
