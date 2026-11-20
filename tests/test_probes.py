import pandas as pd
import pytest
import torch

import tor.data.load
import tor.probes.memtime as mt
from tor.data.metadata import length_percentiles
from tor.probes.memtime import isolated_runner, probe_config, run_plan, run_probe
from tor.probes.shapes import FITS_SHAPE, ProbeShape, probe_grid
from tor.runlog import config_hash, new_record, read_all
import probe_fakes
from probe_fakes import FakeAdapter, NoCkptAdapter


def _meta():
    return pd.DataFrame({
        "name": list("abcdefg"),
        "archive": ["UCR"] * 5 + ["UEA", "UEA"],
        "length": [16, 24, 32, 40, 64, 48, 2000],
        "channels": [1, 1, 1, 1, 1, 6, 6],
        "equal_length": [True] * 7,
        "missing": [False] * 5 + [False, True],
    }).astype({"equal_length": "boolean", "missing": "boolean"})


@pytest.fixture
def fake(monkeypatch):
    class Oom64(FakeAdapter):
        oom_batch = 64

    class CkptOnly(FakeAdapter):
        name = "fake_ckpt"
        oom_without_ckpt = True

    reg = {"fake_probe": FakeAdapter, "fake_oom64": Oom64, "fake_ckpt": CkptOnly,
           "fake_nockpt": NoCkptAdapter}
    monkeypatch.setattr(mt, "get_adapter", lambda n: reg[n])
    return reg


def _cpu_runner(cfg):
    return mt._run_cfg(cfg, "cpu")


def _plan(models, tmp_path, runner=_cpu_runner, modes=("head",), **kw):
    return run_plan(list(models), list(modes), _meta(), runner, tmp_path / "log.jsonl",
                    n_series=128, warmup_epochs=1, timed_epochs=1, **kw)


def test_length_percentiles_require_no_missing():
    m = _meta()
    assert length_percentiles(m)["max"] == 64
    assert length_percentiles(m, require_no_missing=False)["max"] == 2000
    m.loc[0, "equal_length"] = False
    assert length_percentiles(m, require_no_missing=False)["p10"] > 16


def test_grid_size_and_fits_shape():
    meta = _meta()
    grid = probe_grid(meta)
    assert len(grid) <= 24 and len(set(grid)) == len(grid)
    assert {s.channels for s in grid} == {1, 6}
    assert max(s.length for s in grid) == 2000  # missing-value rows count for shape
    p90 = length_percentiles(meta, require_no_missing=False)["p90"]
    assert FITS_SHAPE(meta) == ProbeShape(p90, 1, 8)


def test_oom_recorded_and_grid_continues(fake, monkeypatch):
    events = probe_fakes.EVENTS
    events.clear()
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty"))
    recs = [run_probe("fake_oom64", "head", ProbeShape(40, 2, b), n_series=128,
                      warmup_epochs=1, timed_epochs=1, device="cpu") for b in (8, 32, 64)]
    assert [r.status for r in recs] == ["ok", "ok", "oom"]
    assert events[-2:] == ["oom", "empty"]  # cache emptied after the OOM, not just at start
    x = recs[0].extra
    for k in ("s_per_epoch", "s_per_1000_series", "params_total", "params_trainable",
              "inference_s_per_1000", "epoch_times_s", "unused_params", "device",
              "peak_gpu_mem_reserved_mib"):
        assert k in x
    assert x["unused_params"] == 7 and x["raw_len"] == 40 and x["preprocessed_len"] == 32
    assert len(x["epoch_times_s"]) == 2 and x["grad_ckpt"] is False


def test_oom_in_inference(fake, monkeypatch):
    class InferOom(FakeAdapter):
        oom_infer = True

    monkeypatch.setattr(mt, "get_adapter", lambda n: InferOom)
    r = run_probe("x", "head", ProbeShape(40, 1, 8), n_series=64, warmup_epochs=1,
                  timed_epochs=1, device="cpu")
    assert r.status == "oom" and len(r.extra["epoch_times_s"]) == 2
    assert "inference_s_per_1000" not in r.extra


def test_is_oom_markers():
    for msg in ("CUDA out of memory", "CUBLAS_STATUS_ALLOC_FAILED", "CUDNN_STATUS_ALLOC_FAILED x"):
        assert mt._is_oom(RuntimeError(msg))
    assert not mt._is_oom(RuntimeError("shape mismatch"))


def test_diverged_s_per_epoch_uses_complete_epochs_only(monkeypatch):
    from tor.train import TrainResult

    def fake_train(model, X, y, recipe, epochs, device, amp=True, on_epoch=None):
        return TrainResult([1.0, float("nan")], [0.5, 9.0], True)

    monkeypatch.setattr(mt, "train_epochs", fake_train)
    monkeypatch.setattr(mt, "get_adapter", lambda n: FakeAdapter)
    r = run_probe("x", "head", ProbeShape(40, 1, 8), n_series=16, warmup_epochs=1,
                  timed_epochs=2, device="cpu")
    assert r.status == "diverged" and r.extra["epoch_times_s"] == [0.5]
    assert r.extra["s_per_epoch"] is None  # only the warmup epoch completed
    fake_train2 = lambda *a, **k: TrainResult([1.0, 1.0, float("nan")], [0.5, 1.0, 9.0], True)  # noqa: E731
    monkeypatch.setattr(mt, "train_epochs", fake_train2)
    r = run_probe("x", "head", ProbeShape(40, 1, 8), n_series=16, warmup_epochs=1,
                  timed_epochs=2, device="cpu")
    assert r.extra["s_per_epoch"] == 1.0


def test_ckpt_retry_on_fits_oom(fake, tmp_path):
    new = _plan(["fake_ckpt"], tmp_path, fits_only=True)
    fits = FITS_SHAPE(_meta())
    assert [(r.status, r.config["grad_ckpt"]) for r in new] == [("oom", False), ("ok", True)]
    assert all((r.config["length"], r.config["channels"], r.config["batch"])
               == (fits.length, 1, 8) for r in new)


def test_no_ckpt_support_recorded(fake, tmp_path):
    new = _plan(["fake_nockpt"], tmp_path, fits_only=True)
    assert [r.status for r in new] == ["oom", "not_run: no grad ckpt support"]


def test_unsupported_mode_recorded_once(fake, tmp_path):
    new = _plan(["fake_probe"], tmp_path, modes=("head", "lora"), fits_only=True)
    assert [r.status for r in new].count("not_run: mode unsupported") == 1
    assert _plan(["fake_probe"], tmp_path, modes=("head", "lora"), fits_only=True) == []


def test_full_grid_runs_fits_first_then_rest(fake, tmp_path):
    order = []

    def runner(cfg):
        order.append((cfg["length"], cfg["channels"], cfg["batch"]))
        return _cpu_runner(cfg)

    new = _plan(["fake_probe"], tmp_path, runner=runner, jobs=1)
    fits = FITS_SHAPE(_meta())
    assert order[0] == (fits.length, 1, 8)
    assert len(new) == len(probe_grid(_meta())) and all(r.status == "ok" for r in new)


def test_resume_skips_done_cells(fake, tmp_path):
    from tor.runlog import append
    log = tmp_path / "log.jsonl"
    cfg = probe_config("fake_probe", "head", FITS_SHAPE(_meta()), 128, 1, 1, False)
    append(new_record("old", "probe", cfg, "ok"), log)
    calls = []

    def runner(c):
        calls.append(config_hash(c))
        return _cpu_runner(c)

    assert _plan(["fake_probe"], tmp_path, runner=runner, fits_only=True) == []
    assert calls == [] and len(read_all(log)) == 1
    # error / not_run records are retried, not skipped
    log.write_text("")
    append(new_record("old", "probe", cfg, "error"), log)
    assert len(_plan(["fake_probe"], tmp_path, runner=runner, fits_only=True)) == 1 and calls


def _seed_log(tmp_path, shared):
    from tor.runlog import append
    log = tmp_path / "log.jsonl"
    cfg = probe_config("fake_probe", "head", FITS_SHAPE(_meta()), 128, 1, 1, False)
    append(new_record("old", "probe", cfg, "ok", cost={"shared": shared}), log)
    return log


@pytest.mark.parametrize("shared,expect", [(True, 1), (None, 1), (False, 0)])
def test_rerun_shared(fake, tmp_path, shared, expect):
    log = _seed_log(tmp_path, shared)
    new = _plan(["fake_probe"], tmp_path, fits_only=True, rerun_shared=True)
    assert len(new) == expect and len(read_all(log)) == 1 + expect  # old record kept


def test_force_reruns_everything(fake, tmp_path):
    log = _seed_log(tmp_path, False)
    assert _plan(["fake_probe"], tmp_path, fits_only=True) == []
    assert len(_plan(["fake_probe"], tmp_path, fits_only=True, force=True)) == 1
    assert len(read_all(log)) == 2


def test_cli_gives_children_threads_by_jobs(monkeypatch):
    seen = {}
    monkeypatch.setattr(mt, "isolated_runner", lambda t, device=None, threads=16: seen.update(
        threads=threads) or "r")
    monkeypatch.setattr(mt, "run_plan", lambda *a, **k: [])
    monkeypatch.setattr(mt, "load_metadata", lambda: None)
    mt.main(["--model", "x", "--jobs", "2"])
    assert seen["threads"] == 8
    mt.main(["--model", "x", "--jobs", "5"])  # capped at 2
    assert seen["threads"] == 8
    mt.main(["--model", "x", "--jobs", "1"])
    assert seen["threads"] == 16


def test_child_thread_cap_recorded(monkeypatch):
    monkeypatch.setenv(mt.EXTRA_MODULES_ENV, "probe_fakes")
    monkeypatch.setenv("TOR_PROBE_FAKES_ONLY", "1")
    cfg = probe_config("fake_probe", "head", ProbeShape(40, 1, 8), 16, 1, 1)
    rec = isolated_runner(timeout_s=60, device="cpu", threads=2)(cfg)
    assert rec.status == "ok" and rec.cost["torch_threads"] <= 2


def test_timeout_includes_partial_progress(monkeypatch):
    monkeypatch.setenv(mt.EXTRA_MODULES_ENV, "probe_fakes")
    monkeypatch.setenv("TOR_PROBE_FAKES_ONLY", "1")
    cfg = probe_config("fake_slow", "head", ProbeShape(40, 1, 8), 16, 1, 20)
    rec = isolated_runner(timeout_s=8, device="cpu")(cfg)
    assert rec.status.startswith("not_run: timeout")
    assert len(rec.extra["epoch_times_s"]) >= 1 and "peak_gpu_mem_mib" in rec.extra["partial"]


def test_probe_uses_no_real_data(fake, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("real data touched")

    monkeypatch.setattr(tor.data.load, "load_train", boom)
    monkeypatch.setattr(tor.data.load, "load_test", boom)
    r = run_probe("fake_probe", "full", ProbeShape(40, 1, 8), n_series=64, warmup_epochs=1,
                  timed_epochs=1, device="cpu")
    assert r.status == "ok" and r.extra["params_trainable"] == r.extra["params_total"]


def test_timeout_path(monkeypatch):
    monkeypatch.setenv(mt.EXTRA_MODULES_ENV, "probe_fakes")
    cfg = probe_config("fake_sleep", "head", ProbeShape(40, 1, 8), 64, 1, 1)
    rec = isolated_runner(timeout_s=3, device="cpu")(cfg)
    assert rec.status == "not_run: timeout (3 s)" and rec.config_hash == config_hash(cfg)


def test_child_crash_is_error(monkeypatch):
    monkeypatch.setenv(mt.EXTRA_MODULES_ENV, "no_such_module_xyz")
    cfg = probe_config("fake_sleep", "head", ProbeShape(40, 1, 8), 64, 1, 1)
    rec = isolated_runner(timeout_s=30, device="cpu")(cfg)
    assert rec.status == "error" and "code" in rec.extra["error"]
