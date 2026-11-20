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
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    recs = [run_probe("fake_oom64", "head", ProbeShape(40, 2, b), n_series=128,
                      warmup_epochs=1, timed_epochs=1, device="cpu") for b in (8, 32, 64)]
    assert [r.status for r in recs] == ["ok", "ok", "oom"]
    assert calls
    x = recs[0].extra
    for k in ("s_per_epoch", "s_per_1000_series", "params_total", "params_trainable",
              "inference_s_per_1000", "epoch_times_s", "unused_params", "device"):
        assert k in x
    assert x["unused_params"] == 7 and x["raw_len"] == 40 and x["preprocessed_len"] == 32
    assert len(x["epoch_times_s"]) == 2 and x["grad_ckpt"] is False


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
