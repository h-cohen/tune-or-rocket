"""Mantis adapters: `mantis8m` (paris-noah/Mantis-8M, MantisV1 class) and `mantisv2`
(paris-noah/MantisV2, MantisV2 class). Package: `mantis-tsfm` (PyPI; repo vfeofanov/mantis).

Input / length: both networks take univariate input (n, 1, L) with L a multiple of 32
(the number of patches). MantisV1 fixes seq_len=512 at construction (config.json);
MantisV2 accepts any multiple of 32. Following the repo README ("Prepare Data": resize with
`F.interpolate(..., size=512, mode='linear', align_corners=False)`, since Mantis was
pre-trained on length 512) `preprocess` linearly interpolates every series, shorter or
longer, to 512. Per-series normalisation is done inside the network (TokenGeneratorUnit.ts_scaler).
https://github.com/vfeofanov/mantis#prepare-data

Channels: per channel. (n, C, 512) -> (n*C, 1, 512) -> backbone -> (n*C, 256) -> (n, C*256),
the same scheme as `MantisTrainer.transform` (it loops over `x[:, [i], :]` and concatenates).

Embedding: the network's own output token (`output_token='cls_token'`, last layer), 256-d.

Head: LayerNorm + Linear on the concatenated embedding, as in the paper's fine-tuning
protocol (arXiv 2502.15637v2, Appendix C.1: "The prediction head is a layer normalization
step followed by a linear layer"). (mantis-tsfm's own default head is BatchNorm1d + Linear.)

Scratch: the package class constructed with its default hyper-parameters and a seeded
random initialisation, which is identical to the HF config.json of each repo (checked).

LoRA: peft on the attention projections. MantisV1 (mantis/architecture/transformer_v1_utils/
transformer.py, Attention): fused `to_qkv` and output `to_out.0`. MantisV2
(transformer_v2_utils/attention.py, TimeWiseMultiheadAttention): fused `wQKV` and `wO`.

Gradient checkpointing: not implemented (the 4-8M-parameter models do not need it).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tor.models import register
from tor.models.base import AdapterUnavailable, FMAdapter, Mode, Recipe

try:
    from mantis.architecture import MantisV1, MantisV2
except ImportError as e:  # registry skips the module
    raise AdapterUnavailable(f"mantis-tsfm not installed: {e}") from e

SEQ_LEN = 512
EMB_DIM = 256


def _hub_kwargs() -> dict:
    from tor.config import get_settings, load_secrets

    load_secrets()
    cache = get_settings().cache_dir / "hf"
    cache.mkdir(parents=True, exist_ok=True)
    return {"cache_dir": str(cache)}


class MantisClassifier(nn.Module):
    """Per-channel Mantis backbone + LayerNorm/Linear head. Input (n, C, 512)."""

    def __init__(self, backbone: nn.Module, n_classes: int, n_channels: int):
        super().__init__()
        self.backbone = backbone
        self.n_channels = n_channels
        d = EMB_DIM * n_channels
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        n, c, length = x.shape
        z = self.backbone(x.reshape(n * c, 1, length))
        return z.reshape(n, c * z.shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class _MantisBase(FMAdapter):
    hf_repo: str
    max_length = SEQ_LEN
    multivariate = "per_channel"
    supported_modes: tuple[Mode, ...] = ("head", "lora", "full", "scratch")
    supports_grad_ckpt = False
    _cls: type

    def __init__(self):
        self._backbone: nn.Module | None = None
        self._init_kind = None
        self._pretrained_state: dict | None = None

    # -- construction -------------------------------------------------------
    @classmethod
    def _random_backbone(cls, seed: int) -> nn.Module:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return cls._cls(device="cpu")

    def load(self, init) -> None:
        if init == "pretrained":
            net = self._cls(device="cpu").from_pretrained(self.hf_repo, **_hub_kwargs())
            self._init_kind = "pretrained"
            # kept to build independent copies (deepcopy of MantisV2 fails: rope has a staticmethod)
            self._pretrained_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            kind, seed = init
            if kind != "random":
                raise ValueError(f"bad init {init!r}")
            net = self._random_backbone(seed)
            self._init_kind = "random"
        self._backbone = net.eval()

    # -- data ---------------------------------------------------------------
    def preprocess(self, X: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        if x.ndim == 2:
            x = x[:, None, :]
        return F.interpolate(x, size=SEQ_LEN, mode="linear", align_corners=False)

    def embed(self, X: torch.Tensor) -> torch.Tensor:
        if self._backbone is None:
            raise RuntimeError("call load() first")
        net = MantisClassifier(self._backbone, 1, X.shape[1])  # head unused
        self._backbone.eval()
        with torch.no_grad():
            return net.features(X)

    # -- classifier ---------------------------------------------------------
    def classifier(self, n_classes, mode, n_channels, seq_len, grad_ckpt=False, seed=0):
        self._check_mode(mode, grad_ckpt)
        if seq_len != SEQ_LEN:
            raise ValueError(f"{self.name} expects preprocessed length {SEQ_LEN}, got {seq_len}")
        if mode == "scratch":
            backbone = self._random_backbone(seed)
        else:
            if self._init_kind != "pretrained":
                raise RuntimeError(f"{mode} needs load('pretrained') first")
            backbone = self._cls(device="cpu")
            backbone.load_state_dict(self._pretrained_state)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if mode == "lora":
                import peft

                backbone = peft.get_peft_model(backbone, self.lora_config())
            model = MantisClassifier(backbone, n_classes, n_channels)
        if mode == "head":
            for p in backbone.parameters():
                p.requires_grad = False
        # lora: peft froze the base weights; full/scratch: everything trainable
        return model

    def default_recipe(self) -> Recipe:
        # Paper Appendix C.1: AdamW, lr 2e-4, weight decay 0.05, 500 epochs, batch 128.
        # Warmup: mantis-tsfm's trainer warms up 10 epochs (then cosine; the harness
        # keeps lr constant after warmup), i.e. 10/500 of training.
        return Recipe(lr=2e-4, weight_decay=0.05, epochs=500, batch_size=128, warmup_frac=0.02)

    recipe_source = (
        "Feofanov et al., Mantis, arXiv 2502.15637v2, Appendix C.1 (full fine-tuning, 500 epochs, "
        "batch 128, AdamW lr 2e-4, wd 0.05; LayerNorm+Linear head); 10-epoch warmup from "
        "mantis-tsfm trainer/trainer_utils/scheduling.py"
    )


@register
class Mantis8M(_MantisBase):
    name = "mantis8m"
    hf_repo = "paris-noah/Mantis-8M"
    _cls = MantisV1
    lora_targets = ("to_qkv", "to_out.0")
    pretraining_corpora = (
        "Mantis v1 (Feofanov et al., arXiv 2502.15637): ~1.89M real-world series, the corpus also "
        "used by NuTime (Lin et al. 2024); per the paper (Appendix B.1) this corpus INCLUDES the "
        "UCR and UEA training sets (not strictly OOD)."
    )


@register
class MantisV2Adapter(_MantisBase):
    name = "mantisv2"
    hf_repo = "paris-noah/MantisV2"
    _cls = MantisV2
    lora_targets = ("wQKV", "wO")
    pretraining_corpora = (
        "Mantis v2 (arXiv 2502.15637v2, Section 3.3): pre-trained exclusively on synthetic series "
        "generated by CauKer (Xie et al. 2026), chosen to avoid UCR/UEA leakage; no UCR/UEA data."
    )
