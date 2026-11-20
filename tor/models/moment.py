"""MOMENT adapters: `moment-small`, `moment-base`, `moment-large`
(AutonLab/MOMENT-1-{small,base,large}; Goswami et al., arXiv 2402.03885). Package: `momentfm`
(PyPI 0.1.4; repo moment-timeseries-foundation-model/moment), installed with --no-deps because
its metadata pins transformers==4.33.3 / huggingface-hub==0.24.0 / numpy==1.25.2, which would
downgrade the env; the package imports and runs unchanged on the env's transformers 5.x.

Architecture: T5 encoder (flan-t5-{small,base,large} config), patch length 8, fixed input length
512 (64 patches), RevIN instance normalisation inside the model, per-channel (univariate) processing.

Input / length (`preprocess`): always returns (n, C, 512).
  * shorter series are LEFT-padded to 512 with zeros and an input mask (0 on the pad), following
    the paper ("we sub-sample longer time series, and pad shorter ones with zeros on the left",
    Sec. 3.2) and momentfm/data/classification_dataset.py (`np.pad(ts, (seq_len-len, 0))`,
    `input_mask[:seq_len-len] = 0`).
  * longer series are sub-sampled to 512 points (paper Sec. 3.2). The paper gives no further detail
    and the package code has no routine for it (its classification dataset would fail), so the
    harness picks 512 evenly spaced samples, `round(linspace(0, L-1, 512))`. (The paper evaluated
    only UCR sets shorter than 512, App. F.2.)
  * `forward(x)` takes only the tensor, so the mask is carried inside it: pad positions are NaN
    (`preprocess`), and the model derives `input_mask = ~isnan(x[:, 0])` (all channels share a
    length), then replaces NaN by 0. Real NaNs in the data would be read as padding, so inputs
    must be imputed beforehand. The mask feeds RevIN statistics, the patch-embedding mask and
    the T5 attention mask exactly as in `MOMENT.classify` (models/moment.py).

Channels: per channel. (n, C, 512) -> (n*C, 64, d) through the shared encoder, then
`reduction="concat"` as in `MOMENT.classify` (patch tokens of the C channels concatenated along the
feature axis, d*C), the setting the repo's classification tutorial uses by default
(tutorials/finetune_demo/classification.py `--reduction concat`). Head is sized from n_channels.

Embedding (`embed`): the input to MOMENT's classification head, i.e. `classify()` output
`enc_out` (n, 64, d*C) averaged over patches, `ClassificationHead.forward` (`torch.mean(x, dim=1)`),
giving (n, d*C), d = 512/768/1024. NOTE (MOMENT's own behaviour, reproduced): this mean is over all
64 patches, including fully padded ones; the input mask is not used by the head. (`MOMENT.embed()`
uses a masked mean and channel-averages; it is not used here.)

Head: `momentfm.models.moment.ClassificationHead` (dropout 0.1 + Linear), as in the package.

Scratch: `momentfm.models.moment.MOMENT` built from the repo's config.json with seeded torch RNG;
the T5 encoder is HF `T5EncoderModel(config)` (random HF init) and patch embedding uses the
package default initialisation (orthogonal, gain 1.41). No pretrained weights are loaded.

LoRA: peft on the T5 self-attention projections `q`, `k`, `v`, `o`
(encoder.block.N.layer.0.SelfAttention.{q,k,v,o}); the patch embedder is frozen, head trainable.

Gradient checkpointing: HF `T5Stack.gradient_checkpointing_enable()` on the encoder (the
package itself uses it for pre-training, App. on pre-training, and exposes
`enable_gradient_checkpointing`). Active only in train mode. The package defaults it to True; the
adapter sets it explicitly (off unless `grad_ckpt=True`).

Params: T5Stack keeps an unused `embed_tokens` (vocab x d_model) because the model feeds
`inputs_embeds`; it is part of the checkpoint and of the counts the paper reports (40/125/385M),
but receives no gradient.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import torch
from torch import nn

from tor.models import register
from tor.models.base import AdapterUnavailable, FMAdapter, Mode, Recipe

try:
    from momentfm.models.moment import MOMENT, MOMENTPipeline, ClassificationHead
    from momentfm.utils.masking import Masking
except ImportError as e:  # registry skips the module
    raise AdapterUnavailable(f"momentfm not installed: {e}") from e

SEQ_LEN = 512
PATCH_LEN = 8
_BACKBONE_KW = {
    "task_name": "embedding",  # head = Identity; we attach our own ClassificationHead
    "enable_gradient_checkpointing": False,
    "freeze_embedder": False,
    "freeze_encoder": False,
    "freeze_head": False,
}


def _hub_kwargs() -> dict:
    from tor.config import get_settings, load_secrets

    load_secrets()
    cache = get_settings().cache_dir / "hf"
    cache.mkdir(parents=True, exist_ok=True)
    return {"cache_dir": str(cache)}


class MomentBackbone(nn.Module):
    """RevIN + patching + patch embedding + T5 encoder. (n, C, 512) -> (n, C, 64, d)."""

    def __init__(self, moment: MOMENT):
        super().__init__()
        self.normalizer = moment.normalizer
        self.tokenizer = moment.tokenizer
        self.patch_embedding = moment.patch_embedding
        self.encoder = moment.encoder
        self.d_model = moment.config.d_model
        self.patch_len = moment.patch_len

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        # same operations as MOMENT.classify (momentfm/models/moment.py)
        b, c, _ = x.shape
        x = self.normalizer(x=x, mask=input_mask, mode="norm")
        x = torch.nan_to_num(x, nan=0, posinf=0, neginf=0)
        x = self.tokenizer(x=x)
        enc_in = self.patch_embedding(x, mask=input_mask)
        p = enc_in.shape[2]
        enc_in = enc_in.reshape(b * c, p, self.d_model)
        patch_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attn = patch_mask.repeat_interleave(c, dim=0)
        out = self.encoder(inputs_embeds=enc_in, attention_mask=attn).last_hidden_state
        return out.reshape(b, c, p, self.d_model)


def _split_mask(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """NaN-padded (n, C, L) -> (zero-filled x, float mask (n, L)); mask 0 on the left pad."""
    mask = (~torch.isnan(x[:, 0, :])).to(x.dtype)
    return torch.nan_to_num(x, nan=0.0), mask


def _features(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """concat reduction of MOMENT.classify: (n, C, L) -> (n, 64, C*d)."""
    x, mask = _split_mask(x)
    enc = backbone(x, mask)  # (n, C, P, d)
    n, c, p, d = enc.shape
    return enc.permute(0, 2, 3, 1).reshape(n, p, d * c)


class MomentClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, d_model: int, n_classes: int, n_channels: int):
        super().__init__()
        self.backbone = backbone
        self.head = ClassificationHead(
            n_channels=n_channels, d_model=d_model, n_classes=n_classes, reduction="concat"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(_features(self.backbone, x))


class _MomentBase(FMAdapter):
    hf_repo: str
    max_length = SEQ_LEN
    multivariate = "per_channel"
    supported_modes: tuple[Mode, ...] = ("head", "lora", "full", "scratch")
    supports_grad_ckpt = True
    lora_targets = ("q", "k", "v", "o")
    pretraining_corpora = (
        "Time Series Pile (Goswami et al., arXiv 2402.03885v3, Sec. 3.1 and App. E): 13M series, "
        "1.23B timestamps from 4 public repositories: Informer long-horizon forecasting datasets, "
        "Monash forecasting archive, UCR/UEA classification archive (159 datasets), and TSB-UAD "
        "anomaly benchmark. So UCR/UEA IS included, TRAIN SPLITS ONLY: the paper uses each "
        "creator's splits (60/10/30 random split otherwise, seed 13) and states that MOMENT only "
        "observes the training splits during pre-training. Not verified here: which individual "
        "UCR/UEA datasets entered the Pile (App. E lists only examples)."
    )
    recipe_source = (
        "momentfm repo tutorials/finetune_demo/classification.py (PTB-XL full fine-tuning demo): "
        "Adam, OneCycleLR init lr 1e-6 -> max_lr 1e-4, batch 16, 5 epochs, reduction concat. The "
        "paper (arXiv 2402.03885, App. F.2) publishes no classification fine-tuning recipe (it "
        "evaluates frozen embeddings + SVM); its general fine-tuning text gives batch 64 and a "
        "one-cycle schedule for MOMENT-Large but the peak-lr values did not survive text "
        "extraction. DEVIATIONS: harness holds lr constant after a short warmup (no one-cycle "
        "decay), Adam without weight decay is approximated by AdamW wd=0; 5 epochs is a tutorial "
        "default, not a tuned value"
    )

    def __init__(self):
        self._moment: MOMENT | None = None
        self._init_kind = None
        self._config: dict | None = None

    # -- construction -------------------------------------------------------
    @classmethod
    def _load_config(cls) -> dict:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(cls.hf_repo, "config.json", **_hub_kwargs())
        with open(path) as f:
            return json.load(f)

    def _random_moment(self, seed: int) -> MOMENT:
        if self._config is None:
            self._config = self._load_config()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return MOMENT(copy.deepcopy(self._config), model_kwargs=dict(_BACKBONE_KW))

    def load(self, init) -> None:
        if init == "pretrained":
            net = MOMENTPipeline.from_pretrained(
                self.hf_repo, model_kwargs=dict(_BACKBONE_KW), **_hub_kwargs()
            )
            net.init()
            self._init_kind = "pretrained"
        else:
            kind, seed = init
            if kind != "random":
                raise ValueError(f"bad init {init!r}")
            net = self._random_moment(seed)
            self._init_kind = "random"
        self._moment = net.eval()
        self._backbone = MomentBackbone(net).eval()

    # -- data ---------------------------------------------------------------
    def preprocess(self, X: np.ndarray) -> torch.Tensor:
        x = np.asarray(X, dtype=np.float32)
        if x.ndim == 2:
            x = x[:, None, :]
        n, c, length = x.shape
        if length > SEQ_LEN:  # sub-sample (paper Sec. 3.2)
            idx = np.round(np.linspace(0, length - 1, SEQ_LEN)).astype(int)
            x = x[:, :, idx]
        elif length < SEQ_LEN:  # left pad; NaN marks the pad (see module docstring)
            x = np.concatenate([np.full((n, c, SEQ_LEN - length), np.nan, np.float32), x], axis=-1)
        return torch.as_tensor(np.ascontiguousarray(x))

    def embed(self, X: torch.Tensor) -> torch.Tensor:
        if self._moment is None:
            raise RuntimeError("call load() first")
        self._backbone.eval()
        with torch.no_grad():
            return _features(self._backbone, X).mean(dim=1)

    # -- classifier ---------------------------------------------------------
    def classifier(self, n_classes, mode, n_channels, seq_len, grad_ckpt=False, seed=0):
        self._check_mode(mode, grad_ckpt)
        if seq_len != SEQ_LEN:
            raise ValueError(f"{self.name} expects preprocessed length {SEQ_LEN}, got {seq_len}")
        if mode == "scratch":
            backbone = MomentBackbone(self._random_moment(seed))
        else:
            if self._init_kind != "pretrained":
                raise RuntimeError(f"{mode} needs load('pretrained') first")
            backbone = copy.deepcopy(self._backbone)
        for p in backbone.parameters():
            p.requires_grad = True
        if grad_ckpt:
            backbone.encoder.activation_checkpointing_enable()
        else:
            backbone.encoder.activation_checkpointing_disable()
        d = backbone.d_model
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if mode == "lora":
                import peft

                backbone = peft.get_peft_model(backbone, self.lora_config())
            elif mode == "head":
                for p in backbone.parameters():
                    p.requires_grad = False
            model = MomentClassifier(backbone, d, n_classes, n_channels)
        return model

    def default_recipe(self) -> Recipe:
        return Recipe(lr=1e-4, weight_decay=0.0, epochs=5, batch_size=16, warmup_frac=0.1)


@register
class MomentSmall(_MomentBase):
    name = "moment-small"
    hf_repo = "AutonLab/MOMENT-1-small"


@register
class MomentBase(_MomentBase):
    name = "moment-base"
    hf_repo = "AutonLab/MOMENT-1-base"


@register
class MomentLarge(_MomentBase):
    name = "moment-large"
    hf_repo = "AutonLab/MOMENT-1-large"
