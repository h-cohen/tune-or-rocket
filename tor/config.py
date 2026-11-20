"""Settings and secret handling."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SECRET_KEYS = ("HF_TOKEN", "TABPFN_TOKEN")
DEFAULT_SECRETS_PATH = Path.home() / ".config" / "tune-or-rocket" / "secrets.env"
_DATA_ROOT = "/data/hadar"
_SECRETS: set[str] = set()


@dataclass(frozen=True)
class Settings:
    phase: str
    cache_dir: Path
    repo_root: Path
    runlog_path: Path


def get_settings() -> Settings:
    repo_root = Path(__file__).resolve().parent.parent
    override = os.environ.get("TOR_CACHE")
    if override:
        cache_dir = Path(override)
    elif os.access(_DATA_ROOT, os.W_OK):
        cache_dir = Path(_DATA_ROOT) / "tor-cache"
    else:
        cache_dir = repo_root / "cache"
    return Settings(
        phase=os.environ.get("TOR_PHASE", "0"),
        cache_dir=cache_dir,
        repo_root=repo_root,
        runlog_path=repo_root / "runs/runlog.jsonl",
    )


def load_secrets(path: Path = DEFAULT_SECRETS_PATH) -> None:
    """Load secret keys into os.environ (never overwriting) and remember values for redaction."""
    path = Path(path)
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            if k in SECRET_KEYS and v and k not in os.environ:
                os.environ[k] = v
    for k in SECRET_KEYS:
        if os.environ.get(k):
            _SECRETS.add(os.environ[k])


def redact(s: str) -> str:
    values = set(_SECRETS)
    values.update(os.environ[k] for k in SECRET_KEYS if os.environ.get(k))
    for v in sorted(values, key=len, reverse=True):
        s = s.replace(v, "***")
    return s
