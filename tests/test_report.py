import json
import re

import pandas as pd

from tor import report
from tor.probes.shapes import ProbeShape
from tor.runlog import append, new_record

SHAPE = ProbeShape(100, 1, 8)
META = pd.DataFrame({
    "name": list("abcd"), "archive": ["UCR"] * 3 + ["UEA"], "length": [50, 100, 120, 90],
    "channels": [1, 1, 1, 4], "equal_length": [True] * 4, "missing": [False] * 4,
}).astype({"equal_length": "boolean", "missing": "boolean"})
AVAIL = pd.DataFrame({
    "name": ["Mantis-8M", "UniShape", "Toto"], "family": ["classification", "classification", "forecaster"],
    "available": [True, True, True], "licence": ["apache-2.0", None, "apache-2.0"],
    "notes": [None, "no licence file", None]})


def info(name):
    return {"supported_modes": ("head", "full"), "recipe_source": "R", "pretraining_corpora": "C"}


def probe(model, mode, status="ok", shared=False, s_ep=1.0, ckpt=False, shape=SHAPE, extra=None):
    cfg = {"model": model, "mode": mode, "length": shape.length, "channels": shape.channels,
           "batch": shape.batch, "n_series": 512, "grad_ckpt": ckpt}
    ex = {"s_per_epoch": s_ep, "params_total": 1000, "unused_params": 100,  # used / removed
          "peak_gpu_mem_reserved_mib": 300.0}
    ex.update(extra or {})
    return new_record(f"p-{model}-{mode}", "probe", cfg, status,
                      cost={"shared": shared, "peak_gpu_mem_mib": 200.0}, extra=ex)


def table(recs):
    return report.model_table(recs, AVAIL, SHAPE, info).set_index("model")


def test_missing_shown_not_imputed():
    t = table([])
    assert t.loc["mantis8m", "fits_full"] == "missing: no probe record"
    assert t.loc["mantis8m", "params_total"].startswith("missing")
    assert t.loc["mantis8m", "s_per_epoch_fits (full)"].startswith("missing")


def test_unshared_preferred():
    t = table([probe("mantis8m", "full", shared=True, s_ep=10.0),
               probe("mantis8m", "full", shared=False, s_ep=8.0),
               probe("mantis8m", "full", shared=True, s_ep=12.0)])
    assert t.loc["mantis8m", "s_per_epoch_fits (full)"] == "8.0"
    assert t.loc["mantis8m", "shared_flag"] == "idle"


def test_shared_and_unmeasured_flags():
    t = table([probe("mantis8m", "full", shared=True), probe("toto", "full", shared=None)])
    assert t.loc["mantis8m", "shared_flag"] == "shared"
    assert t.loc["toto", "shared_flag"] == "unmeasured"


def test_fits_statuses_and_ckpt():
    recs = [probe("mantis8m", "full", "oom"), probe("mantis8m", "full", "ok", ckpt=True),
            probe("mantis8m", "head", "oom"), probe("mantis8m", "head", "oom", ckpt=True),
            probe("unishape", "full", "not_run: timeout (5400 s)",
                  extra={"epoch_times_s": [600.0, 650.0]}),
            probe("unishape", "head", "not_run: no idle GPU (x)")]
    t = table(recs)
    assert t.loc["mantis8m", "fits_full"] == "yes (grad ckpt)"
    assert t.loc["mantis8m", "fits_head"] == "no (oom)"
    assert t.loc["mantis8m", "ckpt_needed"] == "full"
    assert t.loc["unishape", "fits_full"] == "timeout (5400 s, partial epochs: 600.0, 650.0)"
    assert t.loc["unishape", "fits_head"] == "not_run: no idle GPU (x)"
    assert t.loc["unishape", "licence"].startswith("none stated")


def test_completed_beats_earlier_timeout():
    t = table([probe("mantis8m", "full", "not_run: timeout (1800 s)"), probe("mantis8m", "full")])
    assert t.loc["mantis8m", "fits_full"] == "yes"


def test_params_used_is_total_minus_unused():
    t = table([probe("mantis8m", "full")])
    # probe params_total = classifier (used); checkpoint total = used + unused
    assert (t.loc["mantis8m", "params_total"], t.loc["mantis8m", "params_unused"],
            t.loc["mantis8m", "params_used"]) == (1100, 100, 1000)
    assert t.loc["mantis8m", "params_source"].startswith("full: ")
    r0 = probe("toto", "full", extra={"unused_params": 0})
    assert table([r0]).loc["toto", "params_unused"] == "0 (adapter reports none)"
    assert table([r0]).loc["toto", "params_total"] == 1000
    assert t.loc["mantis8m", "peak_alloc_mib (full)"] == "200"
    assert t.loc["mantis8m", "peak_reserved_mib (full)"] == "300"


def test_s_per_epoch_only_from_fits_shape():
    other = ProbeShape(120, 1, 8)
    t = table([probe("mantis8m", "full", shape=other, s_ep=99.0)])
    assert t.loc["mantis8m", "fits_full"] == "missing: no probe record"


def test_grid_completeness_counts():
    shapes = [SHAPE, ProbeShape(120, 1, 8)]
    recs = [probe("m", "head"), probe("m", "head", "oom", shape=shapes[1]),
            probe("m", "full", "not_run: timeout (1800 s)")]
    c = report.grid_completeness(recs, {"m": ("head", "full")}, shapes).iloc[0]
    assert (c["planned"], c["ok"], c["oom"], c["timeout"], c["not_run"], c["no_record"]) == (4, 1, 1, 1, 0, 1)


def _log(tmp_path, recs):
    p = tmp_path / "runlog.jsonl"
    for r in recs:
        append(r, p)
    return p


def _render(tmp_path, recs):
    csv = tmp_path / "avail.csv"
    AVAIL.to_csv(csv, index=False)
    return report.render_report(_log(tmp_path, recs), csv, META, info, phase0_dir=tmp_path)


def test_score_blocks_are_excluded_not_masked(tmp_path):
    (tmp_path / "hc2_availability.md").write_text(
        "## Coverage (datasets with a published result / datasets in our metadata)\n\n"
        "| a | b |\n\nGood 0.8621 accuracy.\n\n## Usage restriction\n\nclean text 42\n")
    text = _render(tmp_path, [])
    assert "not embedded (see file)" in text
    assert "0.8621" not in text and "clean text 42" in text


def test_ckpt_needed_requires_decisive_records():
    t = table([probe("mantis8m", "full", "oom"), probe("mantis8m", "full", "ok", ckpt=True)])
    assert t.loc["mantis8m", "ckpt_needed"].startswith("missing")  # head not run


def test_partial_peak_marked_and_flagged():
    r = probe("unishape", "full", "not_run: timeout (1800 s)", extra={
        "epoch_times_s": [1.0], "partial": {"peak_gpu_mem_mib": 777.0, "peak_gpu_mem_reserved_mib": 800.0}})
    r.cost = {}
    t = table([r])
    assert t.loc["unishape", "peak_alloc_mib (full)"] == "\u2265777 (partial, timeout)"
    assert t.loc["unishape", "shared_flag"] == "unmeasured"
    assert t.loc["unishape", "fits_full_run_id"].startswith("p-unishape-full")


def test_ckpt_retry_status_and_latest_nondone():
    t = table([probe("mantis8m", "full", "oom"),
               probe("mantis8m", "full", "not_run: timeout (1800 s)", ckpt=True)])
    assert t.loc["mantis8m", "fits_full"] == "oom; grad-ckpt retry: not_run: timeout (1800 s)"
    a, b = probe("m", "full", "error"), probe("m", "full", "not_run: x", shared=False)
    assert report.pick([a, b]) is b


def test_report_has_no_score_words(tmp_path):
    (tmp_path / "hc2_availability.md").write_text(
        "## Coverage (datasets with a published result / datasets in our metadata)\n\n"
        "| a | b |\n\nNo accuracy value, F1 or val_loss here.\n\n## Usage restriction\n\nfull-label accuracy only\n")
    (tmp_path / "literature.md").write_text("## 1. Search coverage\n\nCould NOT be accessed:\n- X: accuracy 0.9\n\n")
    text = _render(tmp_path, [probe("mantis8m", "full")])
    for w in ("accuracy", "F1", "val_"):
        assert w.lower() not in text.lower()


def test_report_sections_missing_when_no_records(tmp_path):
    text = _render(tmp_path, [])
    assert re.search(r"## 4\. Training-free timing[^\n]*\n\nmissing: not yet run", text)
    assert "missing: not yet run (no tabpfn_limits record)" in text
    assert "missing: deviations.md not found" in text
    assert "## 10. Open items for the user" in text and "UniShape licence" in text


def test_report_renders_baselines_limits_and_deviations(tmp_path):
    (tmp_path / "deviations.md").write_text("# Deviations\n\n- Reverso timeout 5400 s.\n")
    base = new_record("b", "baseline", {"model": "minirocket", "mode": "cpu", "length": 100,
                      "channels": 1, "n_series": 512, "threads": 16}, "ok", cost={"shared": None},
                      extra={"fit_s": 1.5, "transform_s": 0.5, "predict_s": 0.1, "n_features": 9996})
    lim = new_record("l", "tabpfn_limits", {}, "ok", extra={
        "package_version": "9", "client_version": "1", "local": {"v2.5": {"max_samples": 50000,
        "max_features": 2000, "max_classes": 10, "max_cpu_samples": 1000, "reason": None}},
        "hosted": {"max_samples": None, "reason": "not queried"}})
    text = _render(tmp_path, [base, lim])
    assert "minirocket" in text and "9996" in text and "50000" in text
    assert "Reverso timeout 5400 s." in text
    assert "unmeasured" in text


def test_deviations_file_has_required_entries():
    text = (report.PHASE0 / "deviations.md").read_text()
    for kw in ("TOR_CACHE", "5400", "pure-PyTorch", "warmup only", "CUDA context", "warmup pass",
               "88", "ShapeletSim", "partial", "accessible"):
        assert kw in text


def test_probe_grid_table_flattens(tmp_path):
    df = report.probe_grid_table([probe("m", "head", shared=None)])
    assert df.iloc[0]["shared"] == "unmeasured" and df.iloc[0]["length"] == 100
