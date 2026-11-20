"""Cost logger: wall time, peak GPU memory, co-tenancy and load flags."""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

import torch

from tor.gpu import query_gpus

LOAD_THRESHOLD = 48.0


def limit_threads(n: int = 16) -> None:
    """Cap torch CPU threads at n. Callers invoke this explicitly (no import side effect)."""
    if torch.get_num_threads() > n:
        torch.set_num_threads(n)


@dataclass
class CostRecord:
    wall_s: float = 0.0
    peak_gpu_mem_mib: float | None = None
    gpu_model: str | None = None
    cpu_cores: int = 0
    torch_threads: int = 0
    load_start: float = 0.0
    load_end: float = 0.0
    other_pids_start: list[int] = field(default_factory=list)
    other_pids_end: list[int] = field(default_factory=list)
    shared: bool | None = False  # None = co-tenancy could not be measured
    hosted: bool = False
    measure_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _own_tree() -> set[int]:
    """os.getpid() plus all descendants, via ppid in /proc/*/stat."""
    children: dict[int, list[int]] = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as f:
                data = f.read()
            # comm may contain spaces/parens; ppid is the 2nd field after the last ')'
            ppid = int(data.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(name))
    ours, stack = set(), [os.getpid()]
    while stack:
        p = stack.pop()
        if p in ours:
            continue
        ours.add(p)
        stack.extend(children.get(p, []))
    return ours


def _other_pids(gpu_index: int) -> list[int]:
    ours = _own_tree()
    for g in query_gpus():
        if g.index == gpu_index:
            return sorted(p for p in g.other_pids if p not in ours)
    raise ValueError(f"gpu index {gpu_index} not reported by nvidia-smi")


def _torch_device(gpu_index: int) -> int:
    """Torch device ordinal for memory stats.

    Inside a GpuLease CUDA_VISIBLE_DEVICES is the single physical index, so torch
    sees it as cuda:0. Otherwise the physical index is the torch ordinal.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd and "," not in cvd:
        return 0
    return gpu_index


def _try(rec: CostRecord, label: str, fn):
    """Run fn; on failure record the reason in rec.measure_errors and return None."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - measurement must never mask the block's outcome
        rec.measure_errors.append(f"{label}: {type(e).__name__}: {e}")
        return None


@contextmanager
def measure(gpu_index: int | None, hosted: bool = False):
    """Measure a block. gpu_index is the PHYSICAL index (used to match nvidia-smi pids).

    Measurements never raise and never mask an exception from the block; a failed
    measurement leaves its field None/empty and adds a reason to rec.measure_errors.
    rec.shared is True on any evidence of co-tenancy, None if co-tenancy could not be
    measured (and no evidence of sharing was found), else False.
    """
    rec = CostRecord(hosted=hosted)
    rec.cpu_cores = len(os.sched_getaffinity(0))
    use_gpu = gpu_index is not None
    cuda_ok = False
    dev = None
    start_pids = end_pids = None
    if use_gpu:
        dev = _torch_device(gpu_index)
        cuda_ok = bool(_try(rec, "cuda_available", torch.cuda.is_available))
        if not cuda_ok:
            rec.measure_errors.append("cuda unavailable: no peak memory or gpu model")
        start_pids = _try(rec, "pids_start", lambda: _other_pids(gpu_index))
        rec.other_pids_start = start_pids or []
    load_start = _try(rec, "load_start", lambda: os.getloadavg()[0])
    rec.load_start = load_start if load_start is not None else float("nan")
    if cuda_ok:
        _try(rec, "reset_peak", lambda: torch.cuda.reset_peak_memory_stats(dev))
    t0 = time.perf_counter()
    try:
        yield rec
    finally:
        rec.wall_s = time.perf_counter() - t0
        load_end = _try(rec, "load_end", lambda: os.getloadavg()[0])
        rec.load_end = load_end if load_end is not None else float("nan")
        if use_gpu:
            end_pids = _try(rec, "pids_end", lambda: _other_pids(gpu_index))
            rec.other_pids_end = end_pids or []
        n = _try(rec, "torch_threads", torch.get_num_threads)
        rec.torch_threads = n if n is not None else 0
        if cuda_ok:
            peak = _try(rec, "peak_mem", lambda: torch.cuda.max_memory_allocated(dev))
            rec.peak_gpu_mem_mib = None if peak is None else peak / 2**20
            rec.gpu_model = _try(rec, "gpu_model", lambda: torch.cuda.get_device_name(dev))
        evidence = bool(
            rec.other_pids_start
            or rec.other_pids_end
            or (load_start is not None and load_start > LOAD_THRESHOLD)
            or (load_end is not None and load_end > LOAD_THRESHOLD)
        )
        unmeasured = (
            load_start is None
            or load_end is None
            or (use_gpu and (start_pids is None or end_pids is None))
        )
        rec.shared = True if evidence else (None if unmeasured else False)
