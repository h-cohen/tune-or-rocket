"""Phase 0 report generator: measurements only, rendered from the run log and committed files.

Every number comes from a run record or a committed CSV. Anything absent is rendered as
"missing: <reason>", never imputed. The report carries no scores of any kind.

    python -m tor.report [--out reports/phase0.md] [--grid-csv reports/phase0/probe_grid.csv]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Callable

import pandas as pd

from tor.config import get_settings
from tor.probes.shapes import FITS_SHAPE, ProbeShape, probe_grid
from tor.runlog import NOT_RUN_PREFIX, RunRecord, read_all

MODES = ("head", "lora", "full", "scratch")
DONE = ("ok", "oom", "diverged")
PHASE0 = Path(__file__).resolve().parent.parent / "reports" / "phase0"
# Availability-CSV model name -> adapter registry name.
ADAPTER_FOR = {
    "UniShape": "unishape", "Mantis-8M": "mantis8m", "MantisV2": "mantisv2", "NuTime": "nutime",
    "MOMENT-1-small": "moment-small", "MOMENT-1-base": "moment-base",
    "MOMENT-1-large": "moment-large", "Chronos-Bolt-small": "chronos-bolt-small",
    "Chronos-Bolt-base": "chronos-bolt-base", "Chronos-2": "chronos2",
    "TimesFM-2.5": "timesfm25", "Toto": "toto", "Reverso": "reverso",
    "reverso-small": "reverso-small", "InceptionTime": "inceptiontime",
}
_FORBIDDEN = re.compile(r"accuracy|\bF1\b|val_", re.IGNORECASE)


def sanitize(text: str, source: str = "embedded text") -> str:
    """The report is measurement-only. Embedded text containing score vocabulary is excluded
    as a whole block (cited prose is never rewritten)."""
    if _FORBIDDEN.search(text):
        return f"missing: {source} contains score vocabulary; not embedded (see file)"
    return text


def _cfg_values(records: list[RunRecord], key: str) -> str:
    vals = sorted({r.config[key] for r in records if _is_probe(r) and key in r.config})
    return "/".join(str(v) for v in vals) if vals else _missing("no probe record")


def _n_classes() -> str:
    from tor.probes.memtime import N_CLASSES  # constant of the probe runner, not stored per record
    return f"{N_CLASSES} (probe-runner constant)"


def _missing(reason: str) -> str:
    return f"missing: {reason}"


# ------------------------------------------------------------------ record selection

def shared_label(rec: RunRecord) -> str:
    s = rec.cost.get("shared")
    return "shared" if s is True else "idle" if s is False else "unmeasured"


def pick(recs: list[RunRecord]) -> RunRecord | None:
    """Best record for one cell: completed over not_run/error, idle over shared/unmeasured,
    latest among equals; if nothing completed, the latest record."""
    if not recs:
        return None
    done = [r for r in recs if r.status in DONE]
    if not done:
        return recs[-1]  # nothing completed: the latest attempt
    idle = [r for r in done if r.cost.get("shared") is False]
    return (idle or done)[-1]


def _is_probe(r: RunRecord) -> bool:
    return r.kind == "probe"


def _cell(r: RunRecord) -> tuple:
    c = r.config
    return (c.get("model"), c.get("mode"), c.get("length"), c.get("channels"), c.get("batch"),
            bool(c.get("grad_ckpt", False)))


def _index(records: list[RunRecord]) -> dict[tuple, list[RunRecord]]:
    out: dict[tuple, list[RunRecord]] = {}
    for r in records:
        if _is_probe(r) and "length" in r.config:
            out.setdefault(_cell(r), []).append(r)
    return out


def _static_reason(records: list[RunRecord], model: str, mode: str) -> str | None:
    for r in reversed(records):
        if (_is_probe(r) and r.config.get("model") == model and r.config.get("mode") == mode
                and "length" not in r.config and r.status.startswith(NOT_RUN_PREFIX)):
            return r.status
    return None


def _fmt_s(v) -> str:
    return f"{v:.1f}"


def _fits_cell(idx: dict, records: list[RunRecord], model: str, mode: str,
               shape: ProbeShape) -> tuple[str, RunRecord | None]:
    """(fits text, record the timing/memory comes from)."""
    key = (model, mode, shape.length, shape.channels, shape.batch)
    plain = pick(idx.get(key + (False,), []))
    ckpt = pick(idx.get(key + (True,), []))
    if plain is None:
        static = _static_reason(records, model, mode)
        if static:
            return f"not_run: {static[len(NOT_RUN_PREFIX):]}", None
        return _missing("no probe record"), None
    st = plain.status
    if st == "ok":
        return "yes", plain
    if st == "oom":
        if ckpt is not None and ckpt.status == "ok":
            return "yes (grad ckpt)", ckpt
        if ckpt is not None and ckpt.status == "oom":
            return "no (oom)", plain
        if ckpt is not None:
            return f"oom; grad-ckpt retry: {ckpt.status}", plain
        no_ckpt_support = any(r.status == "not_run: no grad ckpt support" and _is_probe(r)
                              and r.config.get("model") == model and r.config.get("mode") == mode
                              for r in records)
        if no_ckpt_support:
            return "no (oom)", plain
        return "no (oom; grad-ckpt retry missing)", plain
    if st == "diverged":
        return "diverged", plain
    if st.startswith(NOT_RUN_PREFIX):
        reason = st[len(NOT_RUN_PREFIX):]
        if reason.startswith("timeout"):
            ep = plain.extra.get("epoch_times_s") or []
            inner = reason[len("timeout"):].strip().strip("()")
            eps = ", ".join(_fmt_s(e) for e in ep) if ep else "none"
            return f"timeout ({inner}, partial epochs: {eps})", plain
        return f"not_run: {reason}", plain
    return "error: " + sanitize(str(plain.extra.get("error", "unknown"))[:120], "error text"), plain


# ------------------------------------------------------------------ model table

def default_adapter_info(adapter: str) -> dict:
    """Class attributes of the registered adapter, or None values with a reason."""
    try:
        from tor.models import get_adapter
        cls = get_adapter(adapter)
    except Exception as e:  # noqa: BLE001 - missing optional dependency, unknown name
        return {"error": f"adapter unavailable ({type(e).__name__})"}
    return {"supported_modes": tuple(cls.supported_modes), "recipe_source": cls.recipe_source,
            "pretraining_corpora": cls.pretraining_corpora,
            "supports_grad_ckpt": bool(getattr(cls, "supports_grad_ckpt", False))}


def _params(idx: dict, model: str, shape: ProbeShape) -> tuple:
    """(checkpoint total, unused, used, source) from the FITS-shape records, full mode preferred.

    Semantics of the probe fields: extra.params_total = count_params(classifier) = parameters
    the classifier actually has (USED); extra.unused_params = checkpoint parameters removed
    from the classifier (0 when the adapter reports none). Checkpoint total = used + unused.
    LoRA records include the adapter parameters, so lora is used only as a last resort."""
    for mode in ("full", "head", "scratch", "lora"):
        for ckpt in (False, True):
            r = pick([x for x in idx.get((model, mode, shape.length, shape.channels, shape.batch,
                                          ckpt), []) if x.extra.get("params_total") is not None
                      and x.status in DONE])
            if r is not None:
                used = int(r.extra["params_total"])
                unused = r.extra.get("unused_params")
                src = f"{mode}{' (includes LoRA adapter params)' if mode == 'lora' else ''}: {r.run_id}"
                if unused is None:
                    return (_missing("unused count not recorded"), _missing("unused count not recorded"),
                            used, src)
                unused = int(unused)
                return (used + unused, f"{unused} (adapter reports none)" if unused == 0 else unused,
                        used, src)
    m = _missing("no probe record")
    return m, m, m, m


def _licence(row: pd.Series) -> str:
    lic = row.get("licence")
    if lic is None or (isinstance(lic, float) and math.isnan(lic)) or str(lic).strip() == "":
        txt = "none stated"
        return txt + " (inclusion is the user's decision)" if row["name"] == "UniShape" else txt
    return str(lic)


def model_table(records: list[RunRecord], availability: pd.DataFrame, fits_shape: ProbeShape | None = None,
                adapter_info: Callable[[str], dict] | None = None) -> pd.DataFrame:
    """One row per model in the availability table. `fits_shape` defaults to FITS_SHAPE of the
    committed metadata. s_per_epoch comes only from FITS-shape probe records."""
    if fits_shape is None:
        from tor.data.metadata import load_metadata
        fits_shape = FITS_SHAPE(load_metadata())
    adapter_info = adapter_info or default_adapter_info
    idx = _index(records)
    rows = []
    for _, a in availability.iterrows():
        name = ADAPTER_FOR.get(a["name"], str(a["name"]).lower())
        info = adapter_info(name)
        fits, recs = {}, {}
        for mode in MODES:
            fits[mode], recs[mode] = _fits_cell(idx, records, name, mode, fits_shape)
        decisive = ("yes", "yes (grad ckpt)", "no (oom)", "diverged")
        supp = [m for m in MODES if m in (info.get("supported_modes") or MODES)]
        undecided = [m for m in supp if fits[m] not in decisive]
        ckpt_modes = [m for m in supp if fits[m] == "yes (grad ckpt)"]
        if undecided:
            ckpt_needed = _missing("no decisive record for " + ", ".join(undecided))
        else:
            ckpt_needed = ", ".join(ckpt_modes) if ckpt_modes else "no"
        full = recs["full"]
        s_ep_txt, flag, src_id = _missing("no ok full-mode FITS record"), "", _missing("no full-mode FITS record")
        mem_a = mem_r = _missing("no full-mode FITS record")
        if full is not None:
            flag = shared_label(full)
            n_rec = len(idx.get((name, "full", fits_shape.length, fits_shape.channels, fits_shape.batch,
                                 bool(full.config.get("grad_ckpt", False))), []))
            src_id = f"{full.run_id} (chosen of {n_rec})"
            if full.status == "ok":
                s_ep = full.extra.get("s_per_epoch")
                s_ep_txt = _fmt_s(s_ep) if s_ep is not None else _missing("no timed epochs")
            part = full.extra.get("partial") or {}
            pa = full.cost.get("peak_gpu_mem_mib")
            pr = full.extra.get("peak_gpu_mem_reserved_mib")
            tag = ""
            if pa is None and part.get("peak_gpu_mem_mib") is not None:
                pa, tag_a = part["peak_gpu_mem_mib"], True
            else:
                tag_a = False
            if pr is None and part.get("peak_gpu_mem_reserved_mib") is not None:
                pr, tag_r = part["peak_gpu_mem_reserved_mib"], True
            else:
                tag_r = False
            mem_a = (f"\u2265{pa:.0f} (partial, timeout)" if tag_a else f"{pa:.0f}") if pa is not None else _missing("peak not measured")
            mem_r = (f"\u2265{pr:.0f} (partial, timeout)" if tag_r else f"{pr:.0f}") if pr is not None else _missing("peak not measured")
        total, unused, used, psrc = _params(idx, name, fits_shape)
        hosted = any(r.cost.get("hosted") for r in records if r.config.get("model") == name)
        modes = info.get("supported_modes")
        rows.append({
            "model": name, "family": a.get("family"), "available": a.get("available"),
            "licence": _licence(a), "params_total": total, "params_unused": unused,
            "params_used": used, "params_source": psrc,
            "modes": ", ".join(modes) if modes else _missing(info.get("error", "no adapter info")),
            "fits_full": fits["full"], "fits_lora": fits["lora"], "fits_head": fits["head"],
            "fits_scratch": fits["scratch"], "ckpt_needed": ckpt_needed,
            "s_per_epoch_fits (full)": s_ep_txt, "shared_flag": flag, "fits_full_run_id": src_id,
            "peak_alloc_mib (full)": mem_a, "peak_reserved_mib (full)": mem_r,
            "hosted": "yes" if hosted else "no",
            "recipe_source": info.get("recipe_source") or _missing("no adapter info"),
            "pretraining_corpora": info.get("pretraining_corpora") or _missing("no adapter info"),
            "notes": "" if pd.isna(a.get("notes")) else str(a.get("notes")),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ grid completeness

def _classify(r: RunRecord | None) -> str:
    if r is None:
        return "no_record"  # planned but absent from the log (not yet run)
    if r.status in ("ok", "oom", "diverged", "error"):
        return r.status
    if r.status.startswith(NOT_RUN_PREFIX + "timeout"):
        return "timeout"
    return "not_run"


def grid_completeness(records: list[RunRecord], models_modes: dict[str, tuple],
                      shapes: list[ProbeShape]) -> pd.DataFrame:
    """Planned cells (supported modes x grid shapes, no grad ckpt) per model vs outcome.
    `no_record` = planned cells with no log entry yet; the outcome columns sum to `planned`."""
    idx = _index(records)
    out = []
    for model, modes in models_modes.items():
        counts = dict.fromkeys(("ok", "oom", "diverged", "timeout", "error", "not_run", "no_record"), 0)
        planned = 0
        for mode in modes:
            for s in shapes:
                planned += 1
                r = pick(idx.get((model, mode, s.length, s.channels, s.batch, False), []))
                counts[_classify(r)] += 1
        out.append({"model": model, "planned": planned, **counts})
    return pd.DataFrame(out)


def probe_grid_table(records: list[RunRecord]) -> pd.DataFrame:
    """Every probe record flattened (appendix CSV)."""
    chosen = {pick(v).run_id for v in _index(records).values()}
    rows = []
    for r in records:
        if not _is_probe(r) or "length" not in r.config:
            continue
        c, e = r.config, r.extra
        rows.append({
            "model": c.get("model"), "mode": c.get("mode"), "length": c.get("length"),
            "channels": c.get("channels"), "batch": c.get("batch"),
            "grad_ckpt": c.get("grad_ckpt"), "status": r.status, "shared": shared_label(r),
            "s_per_epoch": e.get("s_per_epoch"), "s_per_1000_series": e.get("s_per_1000_series"),
            "inference_s_per_1000": e.get("inference_s_per_1000"),
            "peak_alloc_mib": r.cost.get("peak_gpu_mem_mib"),
            "peak_reserved_mib": e.get("peak_gpu_mem_reserved_mib"),
            "params_total": e.get("params_total"), "unused_params": e.get("unused_params"),
            "wall_s": r.cost.get("wall_s"), "gpu": e.get("device"),
            "git_commit": r.git_commit, "run_id": r.run_id, "chosen_for_cell": r.run_id in chosen,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ baselines and limits

def baseline_table(records: list[RunRecord], tabpfn: bool) -> pd.DataFrame:
    """Timing rows of kind=baseline: training-free CPU arms (tabpfn=False) or TabPFN-based arms."""
    rows = []
    for r in records:
        if r.kind != "baseline":
            continue
        is_cpu = r.config.get("mode") == "cpu"
        if is_cpu == tabpfn:
            continue
        e = r.extra
        rows.append({
            "arm": r.config.get("model"), "mode": r.config.get("mode"),
            "length": r.config.get("length"), "channels": r.config.get("channels"),
            "n_series": r.config.get("n_series"), "threads": r.config.get("threads"),
            "status": r.status, "fit_s": e.get("fit_s"), "transform_s": e.get("transform_s"),
            "predict_s": e.get("predict_s"), "n_features": e.get("n_features"),
            "hosted": bool(r.cost.get("hosted")), "shared": shared_label(r),
            "wall_s": r.cost.get("wall_s"),
            "reason": sanitize(str(e.get("reason") or e.get("error") or "")[:160], "reason text")
            if (e.get("reason") or e.get("error")) else "",
        })
    return pd.DataFrame(rows)


def _cell_txt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1000 else f"{v:.0f}"
    return str(v).replace("|", "\\|").replace("\n", " ")


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_cell_txt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _limits_section(records: list[RunRecord]) -> str:
    recs = [r for r in records if r.kind == "tabpfn_limits"]
    parts = []
    if recs:
        lim = recs[-1].extra
        parts.append(f"Limits record `{recs[-1].run_id}` (package tabpfn {lim.get('package_version')}, "
                     f"client {lim.get('client_version')}; read from the installed packages).")
        rows = []
        for ver, d in (lim.get("local") or {}).items():
            rows.append({"where": f"local {ver}", **{k: d.get(k) for k in
                         ("max_samples", "max_features", "max_classes", "max_cpu_samples", "reason")}})
        h = lim.get("hosted") or {}
        rows.append({"where": "hosted (default)", **{k: h.get(k) for k in
                     ("max_samples", "max_features", "max_classes", "reason")}})
        for ver, d in (h.get("by_version") or {}).items():
            rows.append({"where": f"hosted {ver}", **{k: d.get(k) for k in
                         ("max_samples", "max_features", "max_classes")}})
        parts.append(md_table(pd.DataFrame(rows).map(
            lambda v: None if v is None or (isinstance(v, float) and math.isnan(v)) else v)))
        if lim.get("weights_gated"):
            parts.append(f"Weights: {lim['weights_gated']}.")
    else:
        parts.append(_missing("not yet run (no tabpfn_limits record)"))
    t = baseline_table(records, tabpfn=True)
    parts.append("Timing of TabPFN-based arms (synthetic data):")
    parts.append(md_table(t) if len(t) else _missing("not yet run"))
    return "\n\n".join(parts)


# ------------------------------------------------------------------ embedded files

def md_section(text: str, heading: str) -> str | None:
    """Body of the '## <heading>' section (up to the next '## ' heading), or None."""
    m = re.search(rf"^## {re.escape(heading)}\s*$", text, re.MULTILINE)
    if not m:
        return None
    rest = text[m.end():]
    n = re.search(r"^## ", rest, re.MULTILINE)
    return (rest[:n.start()] if n else rest).strip()


def _hc2_section(path: Path) -> str:
    if not path.exists():
        return _missing(f"{path.name} not found")
    text = path.read_text(encoding="utf-8")
    parts = [f"Full notes: `reports/phase0/{path.name}` (sources, name matching). Availability only; "
             "no published result value is read or shown anywhere in this report."]
    for h in ("Coverage (datasets with a published result / datasets in our metadata)",
              "Resample 0 and the official split", "Usage restriction"):
        body = md_section(text, h)
        parts.append(f"**{h}**\n\n" + (sanitize(body, f"section '{h}'") if body else _missing(f"section '{h}' not found")))
    return "\n\n".join(parts)


def _literature_section(path: Path) -> str:
    if not path.exists():
        return _missing(f"{path.name} not found")
    text = path.read_text(encoding="utf-8")
    parts = [f"The literature report is `reports/phase0/{path.name}`. It is not inlined: it quotes "
             "published scores of other methods, and this report carries none. Its open decisions "
             "are listed under 'Open items for the user' below."]
    body = md_section(text, "1. Search coverage")
    m = re.search(r"Could NOT be accessed:\n(.*?)(?:\n\n|\Z)", body or "", re.DOTALL)
    parts.append("Sources that could not be accessed (from the literature report):\n\n" +
                 (sanitize(m.group(1).strip(), "access list") if m else _missing("access list not found")))
    return "\n\n".join(parts)


OPEN_ITEMS = """\
1. UniShape licence: the repository has no licence file (research-use terms unclear). Whether UniShape stays in the study is the user's decision; it is measured here but flagged.
2. Training-free newcomers (TIGER, MASHT, TS2TabPFN, TimEE, ChorusTIC, SwitchPFN) versus the definition of "best training-free pipeline": fixed by preregistration to MR-Hydra and ROCKET+TabPFN with HC2 as reference, or widened to include them (TIGER in particular, which also uses a different tabular model)?
3. TabPFN version: v2.5 local, v3 local, or hosted v3.5. The hosted service caps MASHT at 2000 columns; the choice changes the arms' limits (see TabPFN limits).
4. RocketPFN arXiv ID discrepancy: 2606.21786 in the brief versus 2606.21776 in the literature notes. Unresolved; one check would settle whether they are one paper.
5. UEA HC2 resample convention: that `HC2_TESTFOLDS.csv` resample 0 is the official split is assumed from tsml practice and not verified (UCR is verified).
6. Mantis v1 pretraining corpus size is unreconciled: 7M series (v1 paper, Sec. 2.4) versus 1.89M (v2 text). Name the version whenever a figure is quoted.
"""


# ------------------------------------------------------------------ rendering

def _models_modes(table: pd.DataFrame, availability_modes: dict[str, tuple] | None) -> dict[str, tuple]:
    out = {}
    for _, r in table.iterrows():
        modes = (availability_modes or {}).get(r["model"])
        if modes is None:
            modes = tuple(m.strip() for m in r["modes"].split(",")) if not str(r["modes"]).startswith("missing") else MODES
        out[r["model"]] = modes
    return out


def render_report(runlog: Path | None = None, availability_csv: Path | None = None,
                  meta: pd.DataFrame | None = None, adapter_info: Callable[[str], dict] | None = None,
                  grid_csv_link: str = "phase0/probe_grid.csv", phase0_dir: Path | None = None) -> str:
    runlog = Path(runlog) if runlog is not None else get_settings().runlog_path
    phase0_dir = Path(phase0_dir) if phase0_dir is not None else PHASE0
    availability_csv = Path(availability_csv) if availability_csv else phase0_dir / "models_availability.csv"
    if meta is None:
        from tor.data.metadata import load_metadata
        meta = load_metadata()
    records = read_all(runlog)
    avail = pd.read_csv(availability_csv)
    fits_shape = FITS_SHAPE(meta)
    table = model_table(records, avail, fits_shape, adapter_info)
    shapes = probe_grid(meta)
    comp = grid_completeness(records, _models_modes(table, None), shapes)
    by_kind: dict[str, int] = {}
    for r in records:
        by_kind[r.kind] = by_kind.get(r.kind, 0) + 1
    commits = sorted({r.git_commit[:7] + ("+dirty" if r.git_commit.endswith("+dirty") else "")
                      for r in records if r.git_commit})

    out = ["# Phase 0 report", "",
           "Measurements only: no model-quality score appears anywhere in this report. Every number "
           "is read from a run record (`runs/runlog.jsonl`) or a committed CSV; a missing value is "
           "shown as `missing: <reason>` and is never imputed.", ""]
    out += ["## 1. What was run", "",
            "Run records by kind: " + ", ".join(f"{k} {n}" for k, n in sorted(by_kind.items())) + f" (total {len(records)}).",
            f"Git commits recorded in the log: {', '.join(commits) if commits else _missing('no records')}.",
            f"FITS shape: length {fits_shape.length}, {fits_shape.channels} channel, batch {fits_shape.batch} "
            f"(p90 of lengths over equal-length datasets, including those with unknown missingness: "
            f"require_no_missing=False). Synthetic data; {_cfg_values(records, 'n_series')} series, "
            f"{_cfg_values(records, 'warmup_epochs')} warmup + {_cfg_values(records, 'timed_epochs')} timed epochs "
            f"(from record configs), {_n_classes()} classes.",
            "`s_per_epoch` is the mean over timed epochs of the FITS-shape record. When both idle and "
            "shared (other GPU tenants) records exist, the idle one is used; otherwise the latest "
            "shared or unmeasured record, flagged in `shared_flag`.", ""]
    out += ["## 2. Model table", "",
            md_table(table.drop(columns=["recipe_source", "pretraining_corpora", "notes"])), "",
            "Params: `params_used` is the parameter count of the classifier actually built (probe field "
            "`params_total`); `params_unused` is the checkpoint parameters removed from it (e.g. forecast "
            "decoders; \"0 (adapter reports none)\" means the adapter defines no unused count); "
            "`params_total` here = used + unused = checkpoint size. `params_source` names the mode and "
            "record they came from (full preferred; lora counts include adapter parameters). `fits_*` values: yes, yes (grad ckpt), no (oom), "
            "timeout, not_run: <reason>, missing: <reason>.", "",
            "### Recipes, pretraining corpora and availability notes", ""]
    for _, r in table.iterrows():
        out.append(f"- **{r['model']}** (modes: {r['modes']}; hosted: {r['hosted']}). Recipe: "
                   f"{sanitize(str(r['recipe_source']), 'recipe_source')}. Pretraining corpora (UCR/UEA exposure as stated "
                   f"by the adapter; labels such as inferred/not verified are kept): "
                   f"{sanitize(str(r['pretraining_corpora']), 'pretraining_corpora')}"
                   + (f" Availability note: {sanitize(r['notes'], 'availability note')}" if r["notes"] else ""))
    out += ["", "## 3. Probe grid", "",
            f"Planned cells: supported modes x {len(shapes)} grid shapes per model (no grad-ckpt retries). "
            "`timeout` and `not_run` are cells that were attempted and did not complete (the reason is in "
            "the status); `no_record` cells have no log entry yet. The outcome columns sum to `planned`.", "",
            md_table(comp), "",
            f"Planned total {int(comp['planned'].sum())}; ok {int(comp['ok'].sum())}, oom {int(comp['oom'].sum())}, "
            f"diverged {int(comp['diverged'].sum())}, timeout {int(comp['timeout'].sum())}, "
            f"error {int(comp['error'].sum())}, not_run {int(comp['not_run'].sum())}, "
            f"no_record {int(comp['no_record'].sum())}. The grid is partial unless no_record is 0.", "",
            f"Appendix: every probe record as CSV, [{grid_csv_link}]({grid_csv_link}).", ""]
    tf = baseline_table(records, tabpfn=False)
    out += ["## 4. Training-free timing (CPU, synthetic data)", "",
            md_table(tf) if len(tf) else _missing("not yet run"), ""]
    out += ["## 5. TabPFN limits", "", _limits_section(records), ""]
    smoke = [r for r in records if r.kind == "smoke"]
    sd = sorted({r.config.get("dataset") for r in smoke if r.config.get("dataset")})
    out += ["## 6. Smoke runs", "",
            (f"{len(smoke)} smoke records on dataset(s) {', '.join(sd)} (training split only; "
             "no evaluation of any kind). The dataset is declared pilot-exposed. `mean_epoch_s` = sum of "
             "recorded epoch times / number of epochs (no warmup epoch excluded)."
             if smoke else _missing("not yet run")), ""]
    if smoke:
        rows = []
        for r in smoke:
            e = r.extra
            ep = e.get("epoch_times_s") or []
            rows.append({"model": r.config.get("model"), "mode": r.config.get("mode"),
                         "status": r.status, "epochs": e.get("epochs"),
                         "mean_epoch_s (all epochs incl. first)": sum(ep) / len(ep) if ep else None,
                         "peak_reserved_mib": e.get("peak_gpu_mem_reserved_mib"),
                         "shared": shared_label(r)})
        out += [md_table(pd.DataFrame(rows)), ""]
    out += ["## 7. HC2 and MR-Hydra published-results availability", "",
            _hc2_section(phase0_dir / "hc2_availability.md"), ""]
    out += ["## 8. Literature", "", _literature_section(phase0_dir / "literature.md"), ""]
    dev = phase0_dir / "deviations.md"
    body = dev.read_text(encoding="utf-8").strip() if dev.exists() else _missing("deviations.md not found")
    body = re.sub(r"^# .*\n+", "", body)
    nr = [r for r in records if r.status.startswith(NOT_RUN_PREFIX)]
    reasons: dict[str, int] = {}
    for r in nr:
        reasons[r.status] = reasons.get(r.status, 0) + 1
    out += ["## 9. Deviations and disclosures", "", sanitize(body, "deviations.md"), "",
            "Records with status not_run: " + (", ".join(f"{k} x{v}" for k, v in sorted(reasons.items()))
                                               if reasons else "none") + ".", ""]
    out += ["## 10. Open items for the user", "", OPEN_ITEMS]
    odd = [r for r in records if r.status in ("error", "diverged")]
    out += ["## 11. Anything unexpected", "",
            (f"{len(odd)} records ended in error or diverged: " + "; ".join(
                f"{r.config.get('model')}/{r.config.get('mode')} {r.status}" for r in odd[:20]) + ".")
            if odd else "No error or diverged records in the log.", ""]
    out += ["## 12. Next steps", "",
            "Complete the remaining probe cells (see section 3), resolve the open items above, then "
            "freeze the model list and training-free arms before any scored run.", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runlog", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="write the report here (default stdout)")
    ap.add_argument("--grid-csv", type=Path, default=None, help="probe-grid appendix CSV (default reports/phase0/probe_grid.csv)")
    args = ap.parse_args(argv)
    runlog = args.runlog or get_settings().runlog_path
    grid_csv = args.grid_csv or PHASE0 / "probe_grid.csv"
    grid_csv.parent.mkdir(parents=True, exist_ok=True)
    probe_grid_table(read_all(runlog)).to_csv(grid_csv, index=False)
    text = render_report(runlog)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
