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
    shared: bool = False
    hosted: bool = False

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


def _other_pids(gpu_index: int | None) -> list[int]:
    if gpu_index is None:
        return []
    ours = _own_tree()
    for g in query_gpus():
        if g.index == gpu_index:
            return sorted(p for p in g.other_pids if p not in ours)
    return []


def _torch_device(gpu_index: int) -> int:
    """Torch device ordinal for memory stats.

    Inside a GpuLease CUDA_VISIBLE_DEVICES is the single physical index, so torch
    sees it as cuda:0. Otherwise the physical index is the torch ordinal.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd and "," not in cvd:
        return 0
    return gpu_index


@contextmanager
def measure(gpu_index: int | None, hosted: bool = False):
    """Measure a block. gpu_index is the PHYSICAL index (used to match nvidia-smi pids)."""
    rec = CostRecord(hosted=hosted, cpu_cores=len(os.sched_getaffinity(0)))
    use_gpu = gpu_index is not None
    dev = _torch_device(gpu_index) if use_gpu else None
    rec.other_pids_start = _other_pids(gpu_index)
    rec.load_start = os.getloadavg()[0]
    if use_gpu:
        torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    try:
        yield rec
    finally:
        rec.wall_s = time.perf_counter() - t0
        rec.load_end = os.getloadavg()[0]
        rec.other_pids_end = _other_pids(gpu_index)
        rec.torch_threads = torch.get_num_threads()
        if use_gpu:
            rec.peak_gpu_mem_mib = torch.cuda.max_memory_allocated(dev) / 2**20
            rec.gpu_model = torch.cuda.get_device_name(dev)
        rec.shared = bool(
            rec.other_pids_start
            or rec.other_pids_end
            or rec.load_start > LOAD_THRESHOLD
            or rec.load_end > LOAD_THRESHOLD
        )
