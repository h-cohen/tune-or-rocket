"""Append-only JSONL run log."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

from tor.config import get_settings, redact

KINDS = ("probe", "smoke", "baseline", "tabpfn_limits", "availability")
STATUSES = ("ok", "oom", "diverged", "error")
NOT_RUN_PREFIX = "not_run: "
_BASE_PACKAGES = ("torch", "aeon", "peft", "tabpfn")
_MODEL_PACKAGES = ("momentfm", "chronos", "mantis", "uni2ts", "timesfm")


def config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class RunRecord:
    run_id: str
    kind: str
    config: dict
    config_hash: str
    status: str
    cost: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    git_commit: str = ""
    versions: dict = field(default_factory=dict)
    started: str = ""

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"invalid kind {self.kind!r}; expected one of {KINDS}")
        if self.status not in STATUSES and not self.status.startswith(NOT_RUN_PREFIX):
            raise ValueError(
                f"invalid status {self.status!r}; expected {STATUSES} or '{NOT_RUN_PREFIX}...'"
            )


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=get_settings().repo_root, capture_output=True, text=True, check=True
    ).stdout


def current_git_commit() -> str:
    try:
        commit = _git("rev-parse", "HEAD").strip()
        return commit + ("+dirty" if _git("status", "--porcelain").strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def collect_versions() -> dict[str, str]:
    """Versions of installed packages, read from metadata only (no heavy imports)."""
    names = list(_BASE_PACKAGES) + [m for m in _MODEL_PACKAGES if m in sys.modules]
    out = {}
    for n in names:
        try:
            out[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            pass
    return out


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_record(run_id: str, kind: str, config: dict, status: str, cost: dict | None = None,
               extra: dict | None = None) -> RunRecord:
    """Build a record with config hash, git commit, versions and timestamp filled in."""
    return RunRecord(
        run_id=run_id, kind=kind, config=config, config_hash=config_hash(config),
        status=status, cost=cost or {}, extra=extra or {},
        git_commit=current_git_commit(), versions=collect_versions(), started=now_utc(),
    )


def append(record: RunRecord, path: Path | None = None) -> None:
    path = Path(path) if path is not None else get_settings().runlog_path
    path.parent.mkdir(parents=True, exist_ok=True)
    line = redact(json.dumps(asdict(record), sort_keys=True, default=str)) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def read_all(path: Path | None = None) -> list[RunRecord]:
    path = Path(path) if path is not None else get_settings().runlog_path
    if not path.exists():
        return []
    names = {f.name for f in fields(RunRecord)}
    return [
        RunRecord(**{k: v for k, v in json.loads(line).items() if k in names})
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
