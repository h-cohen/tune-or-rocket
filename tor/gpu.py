"""GPU picker: idle check, flock leases, and a cap on how many GPUs we hold."""
from __future__ import annotations

import fcntl
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from tor.config import get_settings


class NoGpuAvailable(Exception):
    pass


@dataclass
class GpuState:
    index: int
    mem_used_mib: int
    util_pct: int
    other_pids: list[int] = field(default_factory=list)


def _smi(*args: str) -> list[list[str]]:
    out = subprocess.run(
        ["nvidia-smi", *args, "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [[c.strip() for c in line.split(",")] for line in out.splitlines() if line.strip()]


def query_gpus() -> list[GpuState]:
    uuid_to_idx = {u: int(i) for i, u in _smi("--query-gpu=index,uuid")}
    pids: dict[int, list[int]] = {i: [] for i in uuid_to_idx.values()}
    for uuid, pid in _smi("--query-compute-apps=gpu_uuid,pid"):
        if uuid in uuid_to_idx:
            pids[uuid_to_idx[uuid]].append(int(pid))
    return [
        GpuState(int(i), int(mem), int(util), pids.get(int(i), []))
        for i, mem, util in _smi("--query-gpu=index,memory.used,utilization.gpu")
    ]


def is_idle(g: GpuState) -> bool:
    return g.mem_used_mib < 1024 and g.util_pct == 0


def _lock_dir() -> Path:
    d = get_settings().cache_dir / "locks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _try_lock(path: Path):
    """Return an open file holding LOCK_EX, or None if someone else holds it."""
    f = open(path, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def _count_held(d: Path) -> int:
    n = 0
    for p in d.glob("gpu*.lock"):
        f = _try_lock(p)
        if f is None:
            n += 1
        else:
            f.close()  # closing releases the probe lock
    return n


class GpuLease:
    def __init__(self, index: int, fh):
        self.index = index
        self._fh = fh
        self._prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(index)

    def release(self) -> None:
        if self._fh is None:
            return
        if self._prev is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._prev
        self._fh.close()  # releases the flock
        self._fh = None

    def __enter__(self) -> "GpuLease":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def _attempt(d: Path, max_ours: int) -> GpuLease | None:
    # Serialise count+acquire across our processes so the cap cannot be raced.
    with open(d / "pick.mutex", "a+") as m:
        fcntl.flock(m, fcntl.LOCK_EX)
        if _count_held(d) >= max_ours:
            return None
        for g in query_gpus():
            if not is_idle(g):
                continue
            fh = _try_lock(d / f"gpu{g.index}.lock")
            if fh is not None:
                return GpuLease(g.index, fh)
    return None


def acquire_gpu(timeout_s: float = 3600, poll_s: float = 30, max_ours: int = 2) -> GpuLease:
    """Block until an idle, unlocked GPU is leased (cap: max_ours of our leases)."""
    d = _lock_dir()
    deadline = time.monotonic() + timeout_s
    while True:
        lease = _attempt(d, max_ours)
        if lease is not None:
            return lease
        if time.monotonic() >= deadline:
            raise NoGpuAvailable(f"no idle GPU within {timeout_s}s (cap {max_ours})")
        time.sleep(min(poll_s, max(0.0, deadline - time.monotonic())))
