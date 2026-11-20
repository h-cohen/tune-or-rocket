"""Reverso adapters: `reverso` (shinfxh/reverso-base) and `reverso-small` (shinfxh/reverso-small).

Package: github.com/shinfxh/reverso (MIT licence; paper arXiv 2602.17634, Fu et al.), installed with
`pip install --no-deps git+https://github.com/shinfxh/reverso@35f724d3ccf26527f17bc774e897ff0c07d0761a`
(pinned in requirements.lock). No code is copied into this repo. Only the package's torch-native
`reverso_torch` modules are used (pure PyTorch: torch.fft long convolution + a recurrent delta-rule
DeltaNet); the CUDA package `reverso` needs FlashFFTConv and flash-linear-attention (not installed; contract
tests run on CPU). Caveats: (a) `reverso_torch.model.Model` ignores the config key `use_output_pe`, so
it has no `post_pe_q_proj` / `output_position_embedding` that the reverso-base checkpoint contains; both
belong to the forecast decoder, which classifiers do not use. The adapter adds a `post_pe_q_proj` Linear so
that every checkpoint tensor except the sinusoidal buffer `output_position_embedding.pe` (not a parameter)
loads with strict=True. (b) Its DeltaNet is a per-time-step Python loop: slow (2048 steps x 4 layers per
forward), see concerns in the task report. (c) CNNBlock runs the gate/conv input in bfloat16 (package code).

Architecture (HF config.json, equal to the GitHub `checkpoints/reverso/args.json` and
`checkpoints/reverso_small/args.json`): point-wise tokens (no patching), `embedding` Linear(1, d), then
`layers` = [mixer, MLPBlock] x N where mixers alternate `CNNBlock` (long FFT convolution with a learned
length-2048 kernel per channel, gated by a depthwise-conv gate, LayerNorm, residual) and `AttentionBlock`
(`fla`-compatible DeltaNet: q/k/v/o linear projections + short causal convs + beta, 4 heads, LayerNorm,
residual). Forecast decoder (unused by classifiers): `head` Linear(2048, 48), `simple_q_proj`, `key_proj`,
`value_proj`, `out_proj`, and for reverso-base `post_pe_q_proj` + sinusoidal PE.
  reverso       : 8 mixers (conv,attn x4), d_model 128, MLP 512. (README: 2.6M)
  reverso-small : 4 mixers (conv,attn x2), d_model 64,  MLP 256. (README: 550K)
Mapping reverso -> shinfxh/reverso-base is INFERRED: the HF config is byte-identical in content to the GitHub
`checkpoints/reverso/args.json` (the 2.6M model), but the authors do not state the mapping and the HF weights
were not byte-compared with the GitHub LFS file (it could not be fetched; the HF `shinfxh/reverso` repo ships
only reverso_small).

Params (measured by tests/models/test_reverso.py): reverso-base checkpoint 2,609,585 parameters (+ the 268,288-value
non-parameter PE buffer = 2,877,873 checkpoint values) = "2.6M" (README); forecast decoder unused by classifiers
164,529 (head 98,352 + simple_q/key/value 3 x 16,512 + out_proj 129 + post_pe_q_proj 16,512), so the classifier
backbone uses 2,445,056. reverso-small: 550,161 total = "550K"; unused decoder 110,897; used 439,264. Gaps
are by measured subtraction; the README's round figures count the decoder.

Embedding (`embed`) follows Jain et al., arXiv 2611.06988 (HTML fetched via a summarising tool, so details are
not verified against the raw text): Sec. 3 pools the hidden states of all eight layers ("following its
authors' recipe"), Appendix B: width 128 -> 1,024 features per channel (8 x 128 = concatenation of layers; that
reading is INFERRED), context 2,048, series padded at the start with their first value and only real positions
pooled; one step per token, each channel scaled to [0,1] (the model's own min-max normalisation, applied in the
same way here to the padded series). Here: hidden state after each (mixer + MLP) pair (8 for reverso, 4 for
reverso-small), mean over real positions (mean is a HARNESS choice; Jain's Appendix C mentions max pooling,
patch statistics and a differenced-series embedding for the authors' recipe, not reproduced). d = N*d_model
per channel. Per channel independently, embeddings concatenated (d = N*d_model*C).

Classifier head (Jain Sec. 3 / App. B-C, as summarised): last layer, mean and max over real positions (2*d_model
per channel), LayerNorm, zero-initialised Linear. LayerNorm is applied over the concatenation of all channels
(INFERRED; per-channel is not excluded by the text). This differs from `embed` (all layers).

Length (`preprocess` + `_Backbone.pad`): the model needs exactly 2048 points (FFT kernel length, `head`
Linear(2048, 48)). Longer series keep the last 2048 points and shorter series are LEFT-padded with their first
value, as `example/eval_gift.py::ReversoPredictor._left_pad_to_len` and `reverso_torch.forecast.forecast`
(`batch_ctx[:, -seq_len:]`). `preprocess` truncates only; padding happens in the backbone so the real positions
are known and pooling covers real positions only. `preprocess` REJECTS NaN/inf (eval_gift imputes; impute first).
Constant series: package clamps the range to 1e-5, so output is finite.

Scratch: `reverso_torch.model.Model(cfg)` under a seeded RNG: package default init (PyTorch defaults;
`CNNBlock.k` = randn), no pretrained weights.

LoRA: peft on the DeltaNet projections `attention.{q_proj,k_proj,v_proj,o_proj}` (the model has no softmax
attention; the conv mixers have no linear projections besides the 1x1 gate conv). Head trainable.
Gradient checkpointing: torch.utils.checkpoint around every layer (mixer and MLP), training mode only.

Pretraining corpora (paper, as summarised in lit-c notes; not re-verified): GiftEval pretraining data (capped
per dataset) + 1M synthetic series; UCR/UEA not in pretraining per the paper (only the first part of the paper
was read: not fully verified).

Recipe (Jain et al. Sec. 3, App. B/C, as summarised): AdamW, lr 1e-4 network / 1e-3 head, weight decay 0.01,
cosine decay to 0, grad-clip 1.0, bf16 autocast, ~64 channel-series per update, "all weights" fine-tuned,
50 epochs at 1% labels / 20 at 10% (full-label epochs not stated). `Recipe` has one lr and no schedule, so
default_recipe uses lr 1e-4, wd 0.01, batch 64 and epochs=100 (harness default: Jain give no full-label epoch
count); head lr, cosine, clipping are not representable here.
"""
from __future__ import annotations

import copy
import json
import types

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tor.models import register
from tor.models.base import AdapterUnavailable, FMAdapter, Mode, Recipe

try:
    from reverso_torch.model import Model as _Model
except ImportError as e:  # registry skips the module
    raise AdapterUnavailable(f"reverso (torch-native) not installed: {e}") from e

_SEQ = 2048
_DECODER = ("head", "simple_q_proj", "key_proj", "value_proj", "out_proj", "post_pe_q_proj")


def _hub(repo: str, fname: str) -> str:
    from huggingface_hub import hf_hub_download

    from tor.config import get_settings, load_secrets

    load_secrets()
    cache = get_settings().cache_dir / "hf"
    cache.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(repo, fname, cache_dir=str(cache))


def _wrap_checkpoint(block: nn.Module) -> None:
    orig = block.forward

    def fwd(*a, **k):
        if block.training and torch.is_grad_enabled():
            return checkpoint(orig, *a, use_reentrant=False, **k)
        return orig(*a, **k)

    block.forward = fwd


def _build(cfg, seed: int | None = None) -> nn.Module:
    def mk():
        m = _Model(cfg)
        if getattr(cfg, "use_output_pe", False):  # missing in reverso_torch.Model (decoder-only module)
            m.post_pe_q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        return m

    if seed is None:
        return mk()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return mk()


class _Backbone(nn.Module):
    """(n, C, L<=2048) -> per-channel features. `mode`: 'layers' (mean of every layer) or 'last' (mean+max)."""

    def __init__(self, model: nn.Module, features: str):
        super().__init__()
        self.embedding = model.embedding
        self.layers = model.layers
        self.d_model = model.d_model
        self.features = features
        self.n_pairs = len(model.layers) // 2
        self.per_channel_dim = self.n_pairs * self.d_model if features == "layers" else 2 * self.d_model

    @staticmethod
    def pad(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(R, L) -> left-pad with the first value to 2048 (eval_gift `_left_pad_to_len`); bool mask of real points."""
        length = rows.shape[-1]
        mask = torch.ones(rows.shape[0], _SEQ, dtype=torch.bool, device=rows.device)
        if length < _SEQ:
            rows = torch.cat([rows[:, :1].expand(-1, _SEQ - length), rows], dim=-1)
            mask[:, : _SEQ - length] = False
        return rows, mask

    def hidden_states(self, rows: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Per (mixer+MLP) pair hidden states [(R, d, 2048)] and the real-position mask (R, 2048)."""
        x, mask = self.pad(rows)
        x = x.unsqueeze(-1)  # (R, L, 1), as Model.forward
        x_min = x.min(1, keepdim=True)[0].detach()  # use_norm min-max scaling, as Model.forward
        x_range = torch.clamp(x.max(1, keepdim=True)[0].detach() - x_min, min=1e-5)
        h = self.embedding((x - x_min) / x_range).transpose(1, 2)
        out = []
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i % 2 == 1:
                out.append(h)
        return out, mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, length = x.shape
        hs, mask = self.hidden_states(x.reshape(n * c, length))
        w = mask.unsqueeze(1).to(hs[0].dtype)  # (R, 1, L)
        denom = w.sum(-1).clamp(min=1.0)
        if self.features == "layers":
            feats = [(h * w).sum(-1) / denom for h in hs]
        else:
            h = hs[-1]
            feats = [(h * w).sum(-1) / denom, h.masked_fill(~mask.unsqueeze(1), float("-inf")).amax(-1)]
        return torch.cat(feats, dim=-1).reshape(n, c * self.per_channel_dim)


class _Classifier(nn.Module):
    """Jain et al. head: LayerNorm + zero-initialised Linear on last-layer mean/max features."""

    def __init__(self, backbone: nn.Module, n_classes: int, n_channels: int):
        super().__init__()
        self.backbone = backbone
        d = backbone.per_channel_dim * n_channels
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(self.backbone(x)))


class _ReversoBase(FMAdapter):
    max_length = _SEQ
    multivariate = "per_channel"
    supported_modes: tuple[Mode, ...] = ("head", "lora", "full", "scratch")
    supports_grad_ckpt = True
    lora_targets = ("q_proj", "k_proj", "v_proj", "o_proj")
    pretraining_corpora = (
        "Reverso paper (Fu et al., ICML 2026, arXiv 2602.17634): GiftEval pretraining data (capped per dataset) "
        "plus 1M synthetic series (GP kernels, spikes, trend/seasonality/irregularity); UCR/UEA are not in "
        "pretraining per the paper (notes from reading part of the paper only; not fully verified)."
    )
    recipe_source = (
        "Jain et al. arXiv 2611.06988 Sec. 3 + App. B/C (as summarised by a fetch tool, not raw text): AdamW, "
        "lr 1e-4 (head 1e-3), wd 0.01, cosine, clip 1.0, ~64 series/update; epochs for full labels not stated "
        "-> harness default epochs=100; head lr/cosine/clip not representable in Recipe"
    )

    def __init__(self):
        self._model = None
        self._init_kind = None

    @classmethod
    def _cfg(cls):
        return types.SimpleNamespace(**json.load(open(_hub(cls.hf_repo, "config.json"))))

    def load(self, init) -> None:
        cfg = self._cfg()
        if init == "pretrained":
            from safetensors.torch import load_file

            m = _build(cfg)
            sd = load_file(_hub(self.hf_repo, "model.safetensors"))
            sd.pop("output_position_embedding.pe", None)  # sinusoidal buffer of the unused decoder
            m.load_state_dict(sd, strict=True)
            self._init_kind = "pretrained"
        else:
            kind, seed = init
            if kind != "random":
                raise ValueError(f"bad init {init!r}")
            m = _build(cfg, seed)
            self._init_kind = "random"
        self._cfg_obj = cfg
        self._model = m.eval().float()
        self._backbone = _Backbone(self._model, "layers").eval()

    def unused_param_count(self) -> int:
        """Params of the forecast decoder, removed from classifiers."""
        if self._model is None:
            raise RuntimeError("call load() first")
        return sum(p.numel() for n in _DECODER if hasattr(self._model, n) for p in getattr(self._model, n).parameters())

    def preprocess(self, X: np.ndarray) -> torch.Tensor:
        x = np.asarray(X, dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError(f"{self.name}: input contains NaN/inf; impute first")
        if x.ndim == 2:
            x = x[:, None, :]
        if x.shape[-1] > self.max_length:  # package keeps the last seq_len points
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
            model = _build(self._cfg(), seed)
        else:
            if self._init_kind != "pretrained":
                raise RuntimeError(f"{mode} needs load('pretrained') first")
            model = copy.deepcopy(self._model)
        for n in _DECODER:  # forecast decoder, unused
            if hasattr(model, n):
                delattr(model, n)
        for p in model.parameters():
            p.requires_grad = True
        if grad_ckpt:
            for blk in model.layers:
                _wrap_checkpoint(blk)
        backbone = _Backbone(model, "last")
        if mode == "lora":
            import peft

            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                backbone = peft.get_peft_model(backbone, self.lora_config())
            backbone.per_channel_dim = backbone.base_model.model.per_channel_dim
        elif mode == "head":
            for p in backbone.parameters():
                p.requires_grad = False
        return _Classifier(backbone, n_classes, n_channels)

    def default_recipe(self) -> Recipe:
        return Recipe(lr=1e-4, weight_decay=0.01, epochs=100, batch_size=64)


@register
class Reverso(_ReversoBase):
    name = "reverso"
    hf_repo = "shinfxh/reverso-base"


@register
class ReversoSmall(_ReversoBase):
    name = "reverso-small"
    hf_repo = "shinfxh/reverso-small"
