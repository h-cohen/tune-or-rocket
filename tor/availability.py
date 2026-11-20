"""Model availability survey: public weights, licence, gating and size per candidate."""
from __future__ import annotations

import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from huggingface_hub import hf_hub_download, model_info
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    RepositoryNotFoundError,
)

from tor import runlog
from tor.config import get_settings, load_secrets

CSV_COLUMNS = ["name", "family", "hf_repo", "weights_url", "code_url", "paper", "available",
               "licence", "gated", "status", "files_gb", "notes"]
NO_WEIGHTS = "not_run: no public weights found"
GATED = "not_run: gated licence"


@dataclass(frozen=True)
class Candidate:
    name: str
    family: Literal["classification", "general", "forecaster", "deep_ref"]
    hf_repo: str | None
    code_url: str | None
    paper: str
    weights_url: str | None = None  # weights hosted outside Hugging Face
    notes: str = ""


_GH = "https://github.com/"
_RAW = "https://raw.githubusercontent.com/"

CANDIDATES: list[Candidate] = [
    Candidate("UniShape", "classification", None, _GH + "qianlima-lab/UniShape",
              "arXiv 2601.06429",
              weights_url=_RAW + "qianlima-lab/UniShape/main/pretrained_model_ckpt/unishape_checkpoint_finetune.pth",
              notes="No HF repo (HF search empty). Checkpoints in GitHub repo pretrained_model_ckpt/ "
                    "(finetune 56 MB, zeroshot 51 MB). No LICENSE file in repo: licence unknown, "
                    "treat as all rights reserved until clarified."),
    Candidate("Mantis-8M", "classification", "paris-noah/Mantis-8M", _GH + "vfeofanov/mantis",
              "arXiv 2502.15637"),
    Candidate("MantisV2", "classification", "paris-noah/MantisV2", _GH + "vfeofanov/mantis",
              "arXiv 2502.15637 (v2)"),
    Candidate("NuTime", "classification", None, _GH + "chenguolin/NuTime", "arXiv 2310.07402",
              weights_url=_RAW + "chenguolin/NuTime/main/ckpt/checkpoint_bias9.pth",
              notes="No HF repo. Checkpoint ckpt/checkpoint_bias9.pth in GitHub repo."),
    Candidate("MOMENT-1-small", "general", "AutonLab/MOMENT-1-small",
              _GH + "moment-timeseries-foundation-model/moment", "arXiv 2402.03885"),
    Candidate("MOMENT-1-base", "general", "AutonLab/MOMENT-1-base",
              _GH + "moment-timeseries-foundation-model/moment", "arXiv 2402.03885"),
    Candidate("MOMENT-1-large", "general", "AutonLab/MOMENT-1-large",
              _GH + "moment-timeseries-foundation-model/moment", "arXiv 2402.03885"),
    Candidate("Chronos-Bolt-small", "forecaster", "amazon/chronos-bolt-small",
              _GH + "amazon-science/chronos-forecasting", "arXiv 2403.07815"),
    Candidate("Chronos-Bolt-base", "forecaster", "amazon/chronos-bolt-base",
              _GH + "amazon-science/chronos-forecasting", "arXiv 2403.07815"),
    Candidate("Chronos-2", "forecaster", "amazon/chronos-2",
              _GH + "amazon-science/chronos-forecasting", "arXiv 2510.15821"),
    Candidate("TimesFM-2.5", "forecaster", "google/timesfm-2.5-200m-pytorch",
              _GH + "google-research/timesfm", "arXiv 2310.10688 (TimesFM; no separate 2.5 paper)",
              notes="Also flax and transformers variants exist; pytorch variant chosen."),
    Candidate("Toto", "forecaster", "Datadog/Toto-Open-Base-1.0", _GH + "DataDog/toto",
              "arXiv 2505.14766",
              notes="Toto-Open-Base-1.0 (the paper-1 era model). Newer Datadog/Toto-2.0-* "
                    "family exists on HF (4m, 22m, 313m, 1B, 2.5B); not surveyed."),
    Candidate("Reverso", "forecaster", "shinfxh/reverso-base", _GH + "shinfxh/reverso",
              "arXiv 2602.17634",
              notes="2.6M model. GitHub README still says 'coming soon', but HF repo "
                    "shinfxh/reverso-base (updated 2026-04-06) has config 8 layers d_model=128 "
                    "(the paper's 2.6M spec) and an 11.5 MB model.safetensors. Mapping "
                    "reverso-base = 2.6M is inferred from that match, not stated by the authors. "
                    "HF card has no licence; GitHub repo is MIT."),
    Candidate("reverso-small", "forecaster", "shinfxh/reverso-small", _GH + "shinfxh/reverso",
              "arXiv 2602.17634",
              notes="550K model (4 layers, d_model=64). HF card has no licence; GitHub repo is MIT."),
    Candidate("InceptionTime", "deep_ref", None, "https://github.com/hfawaz/InceptionTime",
              "arXiv 1909.04939",
              notes="trained from scratch; no pretrained weights"),
]


def _http(method: str, url: str, headers: dict | None = None) -> tuple[int, dict, bytes]:
    """Minimal HTTP call returning (status, headers, body). Raises URLError on network failure."""
    req = urllib.request.Request(url, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, dict(r.headers), r.read(65536) if method == "GET" else b""
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), b""


def _licence_from_text(text: str) -> str | None:
    t = text.lower()
    if "mit license" in t or "permission is hereby granted, free of charge" in t:
        return "mit"
    if "apache license" in t:
        return "apache-2.0"
    if "gnu general public license" in t:
        return "gpl"
    if "bsd" in t and "redistribution and use" in t:
        return "bsd"
    return "other (see LICENSE)" if t.strip() else None


def _repo_licence(weights_url: str) -> str | None:
    m = re.match(r"https://raw\.githubusercontent\.com/([^/]+/[^/]+)/([^/]+)/", weights_url)
    if not m:
        return None
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt"):
        status, _, body = _http("GET", f"{_RAW}{m.group(1)}/{m.group(2)}/{name}")
        if status == 200:
            return _licence_from_text(body.decode("utf-8", "replace"))
    return None


def _check_url(c: Candidate) -> dict:
    out = dict(available=False, licence=None, gated=False, status="", files_gb=None)
    try:
        status, hdrs, _ = _http("GET", c.weights_url, {"Range": "bytes=0-0"})
    except (urllib.error.URLError, OSError) as e:
        out["status"] = "error"
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    hl = {k.lower(): v for k, v in hdrs.items()}
    if status in (200, 206):
        total = None
        if "content-range" in hl and "/" in hl["content-range"]:
            total = hl["content-range"].rsplit("/", 1)[1]
        elif "content-length" in hl:
            total = hl["content-length"]
        out.update(available=True, status="ok", licence=_repo_licence(c.weights_url),
                   files_gb=round(int(total) / 1e9, 4) if total and total.isdigit() else None)
    else:
        out["status"] = "error"
        out["error"] = f"HTTP {status}"
    return out


def _check_hf(c: Candidate) -> dict:
    out = dict(available=False, licence=None, gated=False, status="", files_gb=None)
    try:
        info = model_info(c.hf_repo, files_metadata=True)
        out["gated"] = bool(info.gated)
        cd = info.card_data
        out["licence"] = getattr(cd, "license", None) if cd is not None else None
        sizes = {s.rfilename: (s.size or 0) for s in (info.siblings or [])}
        if sizes:
            out["files_gb"] = round(sum(sizes.values()) / 1e9, 4)
        cache = str(get_settings().cache_dir / "hf")
        try:
            hf_hub_download(c.hf_repo, "config.json", cache_dir=cache)
        except EntryNotFoundError:
            # No config.json: prove access with the smallest file instead.
            smallest = min(sizes, key=sizes.get) if sizes else None
            if smallest is None:
                raise
            hf_hub_download(c.hf_repo, smallest, cache_dir=cache)
        out.update(available=True, status="ok")
    except GatedRepoError:
        out.update(gated=True, available=False, status=GATED)
    except RepositoryNotFoundError:
        out.update(status="not_run: repo not found")
    except Exception as e:  # network, auth, etc.: recorded, not raised
        out["status"] = "error"
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def check(c: Candidate) -> dict:
    """Probe one candidate. Keys: available, licence, gated, status, files_gb."""
    if c.family == "deep_ref" and not c.hf_repo and not c.weights_url:
        return dict(available=True, licence=None, gated=False, status="ok", files_gb=None)
    if c.hf_repo:
        return _check_hf(c)
    if c.weights_url:
        return _check_url(c)
    return dict(available=False, licence=None, gated=False, status=NO_WEIGHTS, files_gb=None)


def survey(csv_path: Path | None = None, runlog_path: Path | None = None) -> pd.DataFrame:
    load_secrets()
    csv_path = Path(csv_path) if csv_path else (
        get_settings().repo_root / "reports/phase0/models_availability.csv")
    rows = []
    for c in CANDIDATES:
        res = check(c)
        rows.append({**{k: v for k, v in asdict(c).items() if k != "notes"}, **{
            k: res[k] for k in ("available", "licence", "gated", "status", "files_gb")},
            "notes": c.notes})
        extra = {k: res[k] for k in ("available", "licence", "gated", "files_gb")}
        if "error" in res:
            extra["error"] = res["error"]
        runlog.append(runlog.new_record(f"availability-{c.name}", "availability", asdict(c),
                                        res["status"], extra=extra), runlog_path)
    df = pd.DataFrame(rows)[CSV_COLUMNS]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print(survey()[["name", "hf_repo", "available", "licence", "gated", "status", "files_gb"]])
