import os
import subprocess
import sys

import pytest

from tor import gpu
from tor.gpu import GpuState, NoGpuAvailable, acquire_gpu, is_idle


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("TOR_CACHE", str(tmp_path))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def _fake(busy=()):
    return lambda: [
        GpuState(i, 20000 if i in busy else 5, 90 if i in busy else 0, [123] if i in busy else [])
        for i in range(8)
    ]


def test_is_idle_thresholds():
    assert is_idle(GpuState(0, 1023, 0, []))
    assert not is_idle(GpuState(0, 1024, 0, []))
    assert not is_idle(GpuState(0, 10, 1, []))


def test_acquire_picks_idle_unlocked_gpu(monkeypatch):
    monkeypatch.setattr(gpu, "query_gpus", _fake(busy={0, 1, 2, 3}))
    with acquire_gpu(timeout_s=1, poll_s=0.05) as lease:
        assert lease.index == 4
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "4"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_restores_previous_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setattr(gpu, "query_gpus", _fake())
    with acquire_gpu(timeout_s=1, poll_s=0.05):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"


def test_cap_of_two_leases(monkeypatch):
    monkeypatch.setattr(gpu, "query_gpus", _fake())
    with acquire_gpu(timeout_s=1, poll_s=0.05) as a, acquire_gpu(timeout_s=1, poll_s=0.05) as b:
        assert {a.index, b.index} == {0, 1}
        with pytest.raises(NoGpuAvailable):
            acquire_gpu(timeout_s=0.1, poll_s=0.05).__enter__()
    with acquire_gpu(timeout_s=1, poll_s=0.05) as c:
        assert c.index == 0


def test_lock_released_when_holder_process_dies(monkeypatch, tmp_path):
    code = (
        "import os\n"
        "from tor import gpu\n"
        "gpu.query_gpus = lambda: [gpu.GpuState(i, 5 if i == 4 else 20000, 0 if i == 4 else 90, []) for i in range(8)]\n"
        "l = gpu.acquire_gpu(timeout_s=1, poll_s=0.05).__enter__()\n"
        "assert l.index == 4\n"
        "os._exit(0)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], env=dict(os.environ), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    monkeypatch.setattr(gpu, "query_gpus", _fake(busy={0, 1, 2, 3, 5, 6, 7}))
    with acquire_gpu(timeout_s=1, poll_s=0.05) as lease:
        assert lease.index == 4
