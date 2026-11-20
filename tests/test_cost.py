import os
import time

import pytest

from tor import cost
from tor.gpu import GpuState


@pytest.fixture
def fake(monkeypatch):
    state = {"pids": [os.getpid()], "load": 10.0}

    def fake_query():
        return [GpuState(3, 100, 0, list(state["pids"]))]

    monkeypatch.setattr(cost, "query_gpus", fake_query)
    monkeypatch.setattr(cost.os, "getloadavg", lambda: (state["load"], 0.0, 0.0))
    monkeypatch.setattr(cost.torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(cost.torch.cuda, "max_memory_allocated", lambda *a, **k: 5 * 2**20)
    monkeypatch.setattr(cost.torch.cuda, "get_device_name", lambda *a, **k: "Fake GPU")
    return state


def test_shared_false_when_alone(fake):
    with cost.measure(3) as rec:
        pass
    assert rec.shared is False
    assert rec.other_pids_start == [] and rec.other_pids_end == []
    assert rec.peak_gpu_mem_mib == 5.0
    assert rec.gpu_model == "Fake GPU"
    assert rec.cpu_cores == len(os.sched_getaffinity(0))
    assert isinstance(rec.torch_threads, int)
    assert "shared" in rec.to_dict()


def test_shared_true_when_other_pid_appears_at_end(fake):
    with cost.measure(3) as rec:
        fake["pids"] = [os.getpid(), 999]
    assert rec.other_pids_start == []
    assert rec.other_pids_end == [999]
    assert rec.shared is True


def test_shared_true_on_high_load(fake):
    fake["load"] = 50.0
    with cost.measure(3) as rec:
        pass
    assert rec.shared is True


def test_hosted_flag_passthrough(fake, monkeypatch):
    monkeypatch.setattr(cost, "query_gpus", lambda: pytest.fail("no smi expected"))
    with cost.measure(None, hosted=True) as rec:
        pass
    assert rec.hosted is True
    assert rec.peak_gpu_mem_mib is None and rec.gpu_model is None


def test_wall_time_measured(fake):
    with cost.measure(None) as rec:
        time.sleep(0.05)
    assert rec.wall_s >= 0.05
