# SDD ledger — plan: /home/hadar/tune-or-rocket/docs/superpowers/plans/2026-11-20-phase0-harness.md
Spec: docs/superpowers/specs/2026-11-20-phase0-harness-design.md
Branch: phase0-harness (from main 3c40fcc). Implementers: model sonnet (user). Reviewers: sonnet; final review: opus.
Global constraints file: $W/constraints.md

## Pre-flight scan
| Pair / task | Produces vs consumes | Finding |
|---|---|---|
| T1→T2 | redact, get_settings → runlog | ok |
| T2→T8,T10,T11,T12,T14 | RunRecord status `not_run: ...` | ok, prefixes consistent |
| T3→T4,T10,T11 | query_gpus/acquire_gpu | ok |
| T5→T6,T10,T11,T13 | load_train, load_metadata, percentiles | ok |
| T7→T9,T10,T11 | FMAdapter.classifier(n_classes, mode, grad_ckpt) | GAP: no seed for scratch; no supported_modes attr though T7 contract test + T9 Step C rely on per-model supported modes |
| T10→T11,T12,T14 | ProbeShape, FITS_SHAPE | ok |
| T8→T9,T14 | availability CSV | ok |
| T13 | Step 3 parallel research subagents | CONFLICT with implementer no-subagent contract |
| T1 self | tests vs code | ok |
| T2 self | ok |
| T3 self | ok |
| T4 self | torch.set_num_threads at import: side effect | minor, accept |
| T5 self | metadata source "verify" | ok, implementer verifies |
| T6 self | ok |
| T7 self | see gap above |
| T8 self | ok |
| T9 self | multi-family, long, GPU; one implementer | ok, sequential sub-commits |
| T10 self | Step 5 long GPU run | ok, background + wait |
| T11 self | ok |
| T12 self | hosted TabPFN uses external service | approved by user 2026-11-20 |
| T13 self | see conflict |
| T14 self | ok |

## Rulings
- Ruling: branch `phase0-harness` in place, not a worktree — single repo, nothing else in flight — cost if wrong: none, branch can move.
- Ruling: FMAdapter gets class attr `supported_modes: tuple[Mode, ...]` (default all four; InceptionTime ("scratch",)) and `classifier(..., seed: int = 0)` used for scratch init — contract test and T9 need it — cost: tiny interface change.
- Ruling: T13 implementer builds tor/hc2.py + coverage only; literature (Step 3) is run by the controller via research agents — subagent contract forbids nested dispatch — cost: none.
- Ruling: commits must have no Co-Authored-By/session trailers (user rule, overrides template) — told to every implementer.

## Progress
- Task 1: dispatched (BASE 3c40fcc, implementer a134c72832303801a)
- T13 Step 3 literature: controller dispatching 3 research agents in parallel now (outputs to $W/lit-a.md, lit-b.md, lit-c.md; merged into reports/phase0/literature.md during T13)
- lit-c done: R2R ranks confirmed; UniShape 2601.06429, code qianlima-lab/UniShape, weights unconfirmed, no licence; NuTime MIT; Reverso only 550K public (2.6M 'coming soon'); UniShape/NuTime pretrain includes UCR/UEA train. Feed into T8 brief. Numbers via WebFetch summaries — recheck before citing.
- lit-a done: 9 relevant, none high-threat. Medium: UniShape paper 2601.06429 (FT UniShape 0.8708 vs MR-H 0.8621, 128 UCR, p=0.03; scratch ablation 18 ds). Low-med: 2511.23177 motor-fault MOMENT/Mantis FT+LoRA+scratch vs MiniRocket, 1 dataset. Gaps: TS-LIMITS OpenReview, ACML FMLTS, S2/Scholar not searched — follow-up in T13.
- lit-b done: all 3 read. Paper1 = Isiacik et al. ECML PKDD 2025 (4 datasets, FT Mantis/MOMENT vs ROCKET family, no scratch, low-med threat). 2609.07493 none. 2608.06122 low. Isiacik&Ifrim (2610.30950) already in brief.
- Task 1: implementer DONE 9a5fac7; review dispatched (a31dd590040fde469). Note: conda needs CONDA_NO_PLUGINS=true --solver classic.
- Task 1 review: Needs fixes. ⚠️ trailers: controller verified 9a5fac7 author Hadar Cohen <hal.nls@gmail.com>, no trailers — not a gap.
- Task 1: minor (deferred): load_secrets no `export`/inline-comment handling; cache_dir not created by get_settings.
- Task 1: fix round 1/5 (3 addressed, 0 open; commits 9a5fac7..ba3bb3d)
- Task 1: complete (commits 3c40fcc..ba3bb3d, review clean)
- Task 2: dispatched (BASE ba3bb3d, implementer a7e164609d37db6ff). Common implementer rules: implementer-common.md
- Task 2: implementer DONE_WITH_CONCERNS (trailers added despite rule → controller amended message only, tree unchanged, now 4f0e0a6; added .git/hooks/commit-msg stripping trailers). Review dispatched.
- Task 2 review: Needs fixes (dist-name versions; JSON-escaped redaction). ⚠️ runlog in repo → +dirty confirmed real, added as finding 3. Minor O_APPEND promoted to finding 4 (concurrent runs are planned).
- Task 2: minor (deferred): read_all raises on corrupt line; untested new_record/kind validation; git_commit/started default ""; non-str status AttributeError.
- Task 2: fix round 1/5 (4 addressed, 0 open; commits 4f0e0a6..a17f420)
- Task 2: complete (commits ba3bb3d..a17f420, review clean)
- Task 3: dispatched (BASE a17f420, implementer a9ae536ef3b91fdc4)
- Task 3 review: Approved. ⚠️ query_gpus on real node: controller verified 8 entries, uuid→pid mapping populated.
- Task 3: minor (deferred): non-LIFO CUDA_VISIBLE_DEVICES restore (ok: one lease per process by design); nvidia-smi [N/A]/missing errors; no parsing unit test; no cross-process live-lease test; mutex no deadline.
- Task 3: complete (commits a17f420..9e5aa4d, review clean)
- Task 4: dispatched (BASE 9e5aa4d)
- Ruling (T4): replace torch.set_num_threads at import with explicit limit_threads(16) called by entry points — avoids import side effects — cost if wrong: an entry point forgets to call it (T10-T12 dispatches must mention it).
- Task 4 review: Needs fixes (finally masks exception; silent shared=False when unmeasured; CUDA-unavailable). ⚠️ PID namespace: controller verified smi pids exist in host /proc. Minor "descendant pid untested" promoted to finding 4 (core of ruling).
- Ruling (T4): CostRecord.shared becomes bool|None (None = unmeasured) + measure_errors list — explicit-missing rule — cost: T14 report must render None as "unmeasured".
- Task 4: minor (deferred): no tests for pid only-at-start / load_end alone; global os.getloadavg patch; module-level torch import.
- Task 4: fix round 1/5 (4 addressed, 0 open; commits 5b9b314..5e74009)
- Task 4: minor (deferred): NaN load values serialize as non-standard JSON (prefer None); sched_getaffinity/_torch_device unguarded.
- Task 4: complete (commits 9e5aa4d..5e74009, review clean)
- Task 5: dispatched (BASE 5e74009)
- Task 5: implementer DONE_WITH_CONCERNS f1f72af (no RED evidence captured; InsectWingbeatSound absent→NaN; UEA missing=False assumed). Review dispatched. Metadata: 158 rows (128 UCR/30 UEA); p10=68 p50=308 p90=1500 max=17984; rep channels=6.
- Task 5 review: Needs fixes (assumed equal_length/missing for absent row; UEA missing assumed False; guard fail-open).
- Ruling (T5): load_test fail-closed — allowed only for int(TOR_PHASE.strip()) >= 1 — integrity rule — cost: none.
- Ruling (T5): MixedShapesRegularTrain→CSV MixedShapes alias accepted (sizes/length match, disclosed in source_url) — cost if wrong: one metadata row off.
- Task 5: minor (deferred): no-op isinstance branch load.py:36-39; counts stored as floats; percentile fixture test checks only p50/max; no RED evidence captured (process).
- Ruling (T5→T10): probe shapes should not drop UEA rows just because 'missing' is unknown (missingness doesn't change compute shape). In T10, add length_percentiles(meta, require_no_missing: bool = True) param and call it with False for probe_grid. Current (no-missing) values p10=80 p50=315 p90=1496 max=3000 — cost if wrong: probe grid misses the longest UEA lengths (EigenWorms 17984).
- Task 5: fix round 1/5 (4 addressed, 0 open; commits f1f72af..2f34e56)
- Task 5: minor (deferred): convoluted _test_allowed expression; no direct NA-exclusion test for length_percentiles.
- Task 5: complete (commits 5e74009..2f34e56, review clean)
- Task 6: dispatched (BASE 2f34e56)
- Task 6 review: Approved. ⚠️ splits/ committed per plan §6.2 — not a gap.
- Task 6: minor (deferred): label-order test near-vacuous; draw_seed pin not a literal; no exact-k / rounding-tie tests; empty y bare ValueError; non-atomic write_text.
- Task 6: complete (commits 2f34e56..554f577, review clean)
- Task 7: dispatched (BASE 554f577)
- Task 7: implementer DONE_WITH_CONCERNS 0913409 (divergence test via 1e35 inputs; InceptionTime channels from last preprocess; embed RuntimeError before load). Review on opus (load-bearing interface).
- Task 7 review (opus): Needs fixes — inverted fraction in contract test; hidden channel state; broad except; grad_ckpt ignored; head not frozen. Deviations: 1e35 divergence test accepted; embed RuntimeError accepted.
- Ruling (T7): classifier(n_classes, mode, n_channels, seq_len, grad_ckpt=False, seed=0) — FMs need C/L to size heads — cost: plan text for T9-T11 uses old signature; dispatches must state the new one.
- Ruling (T7): adapters get is_available() classmethod, supports_grad_ckpt attr, get_adapter(name); available_adapters catches only ImportError/AdapterUnavailable.
- Task 7: minor (deferred): train_epochs no seed/generator, no y dtype validation; no scratch≠pretrained / seed-determinism contract checks.
- Task 7: fix round 1/5 (6 addressed, 0 open; commits 0913409..8d15c03)
- Task 7: minor (deferred): _freeze_eval skips param-less BN/Dropout next to trainable siblings; get_adapter bare KeyError.
- Task 7: complete (commits 554f577..8d15c03, review clean)
- Ruling (T8): Candidate gets `weights_url: str | None` for non-HF weights (NuTime in GitHub repo, UniShape maybe); reachable → available. Reverso: two rows — reverso (2.6M, brief's model; expected not_run: not public) and reverso-small (550K, public) — cost: one extra row.
- Task 8: dispatched (BASE 8d15c03)
- Task 8: implementer DONE c571415. 15 rows all ok, none gated. UniShape: GitHub weights, NO LICENCE (all rights reserved) → user decision for final report. reverso-base mapped by config match (inferred). runs/runlog.jsonl untracked (plan: committed).
- Task 8 review: Approved.
- Task 8: minor (deferred): CSV drops error reason for error rows; coarse _licence_from_text; case-sensitive LICENSE lookup; files_gb = repo size not loaded weights; notes omit shinfxh/reverso & reverso-nano; no EntryNotFound fallback test.
- Task 8: complete (commits 8d15c03..c571415, review clean). Controller committed runs/runlog.jsonl (data bookkeeping).
- Ruling (T9): split into sub-dispatches per family (9a Mantis, 9b MOMENT, 9c Chronos-Bolt+Chronos-2, 9d TimesFM-2.5, 9e Toto, 9f Reverso base+small, 9g UniShape, 9h NuTime), each its own implementer+review — reviewable diffs, failures isolated — cost: more dispatches.
- Ruling (T9): UniShape adapter is built and measured (public checkpoint) but flagged "no licence file; research-use terms unclear" — inclusion in the study is the user's decision — cost if wrong: wasted adapter.
- Task 9a (Mantis): dispatched (BASE e6fc4aa, implementer a15e6a1724f601d71). Shared T9 instructions: task-9-common.md
- Task 9a: implementer DONE_WITH_CONCERNS 29d20b4 (pandas 3.0.6→2.3.3 downgrade; __init__ module list edit accepted — required for registration).
- Task 9a review: Approved; ⚠️ corpus claim stated as fact but inferred → confirmed real gap (integrity) → fix round 1 sent (+recipe_source cosine/batch note, head-freeze assertions).
- Task 9a: fix round 1/5 (3 addressed, 0 open; commits 29d20b4..0d8bcf0)
- Task 9a: minor (deferred): Mantis v1 corpus 1.89M (v2 text) vs 7M (v1 Sec 2.4) unreconciled; recipe cites v2 for v1 checkpoint; cosine decay not in harness.
- Task 9a: complete (commits e6fc4aa..0d8bcf0, review clean). Params: mantis8m 8,103,936; mantisv2 4,188,672.
- Task 9b (MOMENT): dispatched (BASE 0d8bcf0)
- Task 9b: implementer DONE_WITH_CONCERNS 5214e97 (momentfm --no-deps; params small 35.3M base 109.6M large 341.2M incl ~16-33M unused embed_tokens vs paper 40/125/385M; NaN padding carries mask; >512 subsample is harness choice).
- Task 9b review: Approved + Important (param-count presentation; unused embed_tokens trainable) → fix round 1 sent (+NaN-input guard, mask test). ⚠️ logits==MOMENTPipeline not in tests; citations unfetched — hedged, accept.
- Task 9b: minor (deferred): logit equivalence not tested; loose param bounds; deepcopy of backbone per classifier call (~1.4GB transient for large).
- Ruling (T9b): accept embed_tokens=None in classifiers instead of requires_grad=False — never called (inputs_embeds fed), keeps contract fraction==1.0, saves memory — cost: none.
- Task 9b: fix round 1/5 (3 addressed, 0 open; commits 5214e97..bff1070)
- Task 9b: complete (commits 0d8bcf0..bff1070, review clean). Effective full trainable (1ch,3cls): small 18,889,411; base 84,963,459; large 308,344,323; totals incl unused 35.3/109.6/341.2M.
- Task 9c (Chronos-Bolt s/b + Chronos-2): dispatched (BASE bff1070)
- Task 9c: implementer DONE_WITH_CONCERNS e02e3b8 (no RED; Jain et al pooling unverified beyond abstract; Bolt corpus inferred). Totals bolt-s 47.7M, bolt-b 205.3M, chronos2 119.5M; unused removed 27.7/117.9/3.65M.
- Task 9c review: Needs fixes — pooling includes [REG] contrary to cited Jain App. B.
- Ruling (T9c): pool patch tokens only (exclude [REG]) for Bolt and Chronos-2, mean pooling — match the cited protocol; consistency — cost: none.
- Task 9c: minor (deferred): no model-specific grad-ckpt test; no RED evidence.
- Task 9c: fix round 1/5 (1 addressed, 0 open; commits e02e3b8..c1f309e)
- Task 9c: minor (deferred): pooling not mask-weighted (padded/NaN patches would be averaged) — irrelevant while preprocess truncates & rejects NaN.
- Task 9c: complete (commits bff1070..c1f309e, review clean)
- Task 9d (TimesFM-2.5): dispatched (BASE c1f309e)
- Task 9d: implementer DONE_WITH_CONCERNS 1247de2 (timesfm 3.0.2 --no-deps; total 231.3M vs name 200M unexplained; unused 32.8M; no RED).
- Task 9d review: Needs fixes (doc: param gap = forecast heads by subtraction). Fix round 1 sent (+short/constant input tests).
- Task 9d: fix round 1/5 (2 addressed, 0 open; commits 1247de2..274d22f)
- Task 9d: minor (deferred): __init__ module list long line.
- Task 9d: complete (commits c1f309e..274d22f, review clean). timesfm25 total 231,289,280; unused 32,768,000; used 198.5M.
- Task 9e (Toto): dispatched (BASE 274d22f)
- Task 9e: implementer DONE_WITH_CONCERNS d361ccd (toto-ts --no-deps + gluonts/lightning/jaxtyping/rotary; load_from_checkpoint; total 151,306,080, unused 37,871,712; max_length 4096 from README not verified; paper half-read).
- Task 9e review: Approved (batch independence verified empirically, Δ=0).
- Task 9e: minor (deferred): no batch-independence regression test; classifier builds model twice (deepcopy fails on gluonts scaler).
- Task 9e: complete (commits 274d22f..d361ccd, review clean). toto total 151,306,080; unused 37,871,712. Stopped implementer's lingering waiter.
- Task 9f (Reverso base+small): dispatched (BASE d361ccd)
- Task 9f: implementer DONE_WITH_CONCERNS affdedd (reverso_torch pure-Python DeltaNet loop — slow; timing reflects reference impl, not FlashFFTConv/fla kernels → note in report; base mapping still inferred; Jain details via summarising fetch).
- Task 9f review: Approved.
- Task 9f: minor (deferred → T14 report must state): Reverso timings reflect pure-PyTorch reference impl (Python-loop DeltaNet), not FlashFFTConv/fla kernels; inputs padded to 2048 so compute is length-independent. Docstring points to non-durable task report; Recipe lacks head-lr (Jain: head 1e-3).
- Task 9f: complete (commits d361ccd..affdedd, review clean). reverso total 2,609,585 (used 2,445,056); reverso-small 550,161 (used 439,264).
- Task 9g (UniShape): dispatched (BASE affdedd)
- Task 9g: implementer hit API rate limit mid-task; resumed after reset (uncommitted WIP present).
- Task 9g: implementer DONE_WITH_CONCERNS f0021b7 (no resize in their FT script → native length; scale_len=5 default; per_channel; stubs fastai.basics & timm trunc_normal_; params 3,133,203 total, unused 565,634; clone pinned 78d3265; recipe lr1e-3 wd0.05 300ep bs512).
- Task 9g review: Needs fixes (stubs leak into sys.modules → find_spec ValueError; is_available docstring overclaims hash check). Fix round 1 sent. Note for user (licence decision): _Backbone.forward re-implements ~8 lines of UniShapeModel.forward (needed: theirs requires labels/computes loss); equality-tested.
- Task 9g: minor (deferred): _no_cuda_tensor global patch not thread-safe; unbounded per-length model cache in embed; 16-point minimum justification unverified.
- Task 9g: fix round 1/5 (3 addressed, 0 open; commits f0021b7..920593a)
- Task 9g: complete (commits affdedd..920593a, review clean). unishape total 3,133,203; unused 565,634. LICENCE: none — user decision.
- Task 9h (NuTime): dispatched (BASE 920593a)
- Task 9h: implementer DONE_WITH_CONCERNS 536982d (params 2,395,586 total/2,395,328 used; recipe = Epilepsy demo (bs 2048) vs train_ft.json; corpus claim via notes + 'ucr-merged' in train_ssl.json).
- Task 9h review: Approved. Minor 1 (corpus claim stated flatly) + Minor 2 (cosine not disclosed) treated as integrity gaps → fix round 1. Killed implementer's self-matching pgrep waiter loops (4) and stopped its task before resuming.
- Task 9h: minor (deferred): git rev-parse on every is_available call.
- Cross-cutting (→ final review/T14): several recipes cite cosine schedules; harness trainer is warmup-only — report must state this.
- Task 9h: fix round 1/5 (2 addressed, 0 open; commits 536982d..466ce87)
- Task 9h: complete (commits 920593a..466ce87, review clean). nutime total 2,395,586; used 2,395,328.
- Task 9 (all families): complete (e6fc4aa..466ce87).
- Ruling (T10): split into 10a (code+tests, implementer) and 10b (controller launches the probe CLI as a background job and monitors; results are data, committed by controller). Run order: FITS_SHAPE for every adapter×mode first (report-critical), then the rest of the grid; per-run timeout 1800 s → status "not_run: timeout (1800 s)". Max 2 concurrent processes (one GPU each). — avoids an implementer idling for hours / self-matching waiters — cost: none.
- Ruling (T10, from T5): length_percentiles(meta, require_no_missing: bool = True); probe_grid/FITS_SHAPE use require_no_missing=False (missingness doesn't change compute shape).
- Task 10a: dispatched (BASE 466ce87)
- Task 10a: implementer DONE_WITH_CONCERNS acc6584 (no RED; slow suite not completed in tool time; timeout clock starts after lease).
- Task 10a review (opus): Needs fixes — 32 threads with jobs=2; shared records not re-runnable. Fix round 1 sent incl. minors 2,3,5,6,8 (cheap, measurement quality).
- Task 10a: minor (deferred → T14 report notes): peak excludes CUDA context (~0.5GB); real-dataset peak > probe peak by (N−512)·C·L·4 bytes since train_epochs moves all X to device; inference has no warmup pass; config_hash excludes git commit/device (intended; --force covers).
- 10b plan: run one Reverso FITS cell alone first to size its timeout (scheduling risk).
- Task 10a: fix round 1/5 (7 addressed, 0 open; commits acc6584..d329884)
- Task 10a: complete (commits 466ce87..d329884, review clean)
- Task 10b: starting — Reverso FITS-only sizing run first (jobs 1).
- Task 10b: Reverso FITS run launched in background (bg bafxwgux6), log /data/hadar/tor-cache/logs/probe-reverso-fits.log. Only GPU 7 idle at launch (node busy).
- Ruling (T11): same split — 11a code+tests by implementer (no real GPU runs), 11b real smoke runs launched by controller — cost: none.
- Task 11a: dispatched (BASE d329884)
- Task 11a: implementer DONE_WITH_CONCERNS 6feed53 (smoke dataset = ShapeletSim, n_train 20; memtime gained run_cfg/kind params).
- Task 11a review: Approved.
- Task 11a: minor (deferred): ShapeletSim n_train=20 → 1 step/epoch, "last<first" may flag healthy runs (read losses); test_smoke_diverged accepts error; never-reads-test guard patches an unused name (structurally safe: load_test never imported); vacuous conditional assert; padding='same' warning; no smoke isolated_runner test.
- Task 11a: complete (commits d329884..6feed53, review clean)
- Ruling (T12): same split — 12a code+tests by implementer; 12b timing runs launched by controller (CPU baselines respect 16-core cap minus probe threads; run after/between GPU probes).
- Task 12a: dispatched (BASE 6feed53)
- 10b progress: reverso head FITS ok, 163 s/epoch (batch 8, 512 series), peak reserved 180 MiB, shared=True. Full/lora/scratch may exceed 1800 s timeout (backward through Python loop) — watch.
- Task 12a: implementer DONE_WITH_CONCERNS f6c3387 (tabpfn 10.2.0, client 0.10.1; MASHT n_estimators 8 vs paper '88' uncertain; hosted v3 2000-col cap; papers via summarizer).
- 10b: reverso lora FITS timed out at 1800 s (epochs 613, 690 s; peak 3.1 GB). Ruling: Reverso models get --timeout 5400 (lora epoch ~650 s × 4 + inference ≈ 2800 s) — deviation from the default 1800 s, recorded in each record's timeout status and to be disclosed in the report — cost: ~1.5 GPU-h per reverso mode. Timed-out cells are retried automatically on resume.
- Task 12a review: Approved. Minors 1 (broad not_run matching hides failures), 2 (n_estimators paper value machine-readable), 3 (token scrub at source) promoted → fix round 1.
- Task 12a: minor (deferred): MASHT 10000 budget fails on hosted v3 2000-col cap (flag up front); limits dict for MASHT documentation; long lines/noqa lambdas.
- Task 12a: fix round 1/5 (3 addressed, 0 open; commits f6c3387..8d7404d). Controller ran tests/test_baselines.py to confirm.
- Task 12a: complete (commits 6feed53..8d7404d, review clean)
- Task 13: dispatched implementer (hc2.py + hc2_availability.md + merge lit-a/b/c/d into reports/phase0/literature.md) and a research agent for the literature gaps (lit-d.md: TS-LIMITS OpenReview, ACML FMLTS, Semantic Scholar). BASE 8d7404d
- lit-d done: no new medium/high threat. UTICA 2603.01348 low-medium (FT on UCR/UEA, no ROCKET). TIGER 2610.06156 training-free 0.884 vs HC2 0.876 on 142 UCR → user decision whether it's 'best training-free'. OpenReview/ACML/S2 unreachable (S2 429).
- Task 13 (code): implementer DONE_WITH_CONCERNS bdb9f08. UCR HC2 112/128, MR-H 112/128; UEA HC2 26/30, MR-H 0/30; 30 resamples; resample0=official only per aeon docstring.
- 10b: launched FITS-only probes for the 13 non-Reverso adapters sequentially (bg b6ili5iq1, log /data/hadar/tor-cache/logs/probe-fits-others.log), alongside the Reverso FITS run (bg bafxwgux6). 2 concurrent processes max (one GPU each).
- Ruling (10b scope): full 24-shape grid × 15 adapters × 4 modes ≈ 1,400 cells (days on 2 shared GPUs). Order: FITS for all (report-critical) → Reverso re-run with --timeout 5400 → rest of grid as time permits; the report states exactly which cells ran (counts) and that the grid is partial if so — cost: incomplete grid, disclosed.
- Task 13 review: Approved; reviewer verified UCR resample 0 = official split from Bake off redux (2304.13029). Report corrections (Fungi mischaracterised; add primary-source quote) → fix round 1.
- Task 13: minor (deferred): __main__ read_csv no timeout/error message; no explicit no-accuracy-columns test.
- Task 13: fix round 1/5 (2 addressed, 0 open; commits bdb9f08..08eb4ad). Re-review done by controller reading the doc-only diff (both items addressed) — deviation from dispatching a re-reviewer, justified: 2-sentence doc change.
- Task 13 (code+HC2): complete (commits 8d7404d..08eb4ad). Literature merge dispatched to a writer agent.
- Literature report committed 09277ad (37 rows). Note: it says UniShape weights unverified — actually verified (sha256 + strict load in T9g); RocketPFN ID 2606.21786 vs notes' 2606.21776 unresolved (arXiv API check inconclusive) — carry both into T14 report "open items".
- Task 14a (report code): dispatched (BASE 09277ad); render after probe/smoke/baseline data exist.
- Task 14a: implementer DONE_WITH_CONCERNS 0baaf0d (sanitize() masks score words in embedded text; TOR_CACHE reason missing — reason: / had 63 GB free vs /data 8 TB).
- Task 14a review (opus): Needs fixes — CRITICAL params semantics inverted (negative used counts); deviations missing reasons/items; sanitize masks instead of excluding; recipe_source truncated; ckpt_needed fills unmeasured; partial peaks unlabelled; record choice invisible. Fix round 1 sent (incl. minors 7-13).
- 10b: FITS done for all 13 non-Reverso adapters: every supported mode ok (no OOM at FITS). Deviation (→ T14): the two earlier concurrent CLIs each ran jobs=1 (16 threads each) → up to 32 intra-op threads for FITS records of those runs; from now one CLI at a time with --jobs 2 (8 threads/child).
- 10b: driver launched (bg bo3dzczqz, /data/hadar/tor-cache/logs/driver.sh): reverso & reverso-small FITS @5400 s → smoke all → baselines CPU → hosted TabPFN → rerun-shared FITS → full grid per non-Reverso model.
- Task 14a: implementer fix round 1 done f65e090; re-review next.
- Task 14a: fix round 1/5 (12 addressed, 1 open — params total includes head, mislabelled as checkpoint size; commits 0baaf0d..f65e090). Fix round 2 sent.
- Task 14a: fix round 2/5 (1 addressed, 0 open; f65e090..600d7c7; controller re-checked label-only diff).
- Task 14a: complete (commits 09277ad..600d7c7). Pending: add the "two concurrent CLIs → up to 32 threads in early FITS records" deviation to reports/phase0/deviations.md at render time; render + commit reports/phase0.md and probe_grid.csv when driver finishes (Task 14b).
- Driver killed by Claude Code (system low on memory — other users' jobs use ~180 GB RAM). Completed: reverso FITS step partly. Not run: reverso-small FITS, smoke, baselines, hosted TabPFN, rerun-shared, rest of grid. Run log committed. Waiting for user before restarting (per harness notice).
