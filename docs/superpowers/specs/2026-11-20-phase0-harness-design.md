# Phase 0 and core harness: design

Date: 2026-11-20. Source brief: `NEXT-PAPER-BRIEF.md` (read it first; this document
narrows it to Phase 0 and the harness Phase 0 runs on).

## 1. Intent

Paper 2 asks when fine-tuning a time-series foundation model (FM) beats the best
training-free ROCKET-family pipeline for classification, how much of any gain is due to
pretraining, and at what cost. Before any experiment is designed in detail, Phase 0
must establish three facts, with no accuracy scores:

1. What the literature already shows (re-search; read three unread papers in full).
2. Which candidate FMs can actually be fine-tuned here: public weights, licence, size, a
   working code path, peak GPU memory and time per epoch for head-only, LoRA and full
   fine-tuning.
3. Limits of the ROCKET+TabPFN arm and availability of published HC2 results on the
   archive splits.

The deliverable is a report the steering agent uses to size Phase 1. The code built for
it is the real harness for later phases, not throwaway probes, so the costs measured in
Phase 0 are costs of the code that will run the study.

## 2. Decisions already made (by the user)

- **Standalone paper.** Paper 1's code and embeddings are not available on this machine.
  Everything is built in this repository. Paper 1 results appear only as cited prior
  work; nothing from paper 1 is reused as data. Frozen-embedding arms (brief arms 6–7)
  will be recomputed in later phases.
- **Machine.** All runs on `hpc-node9`: 8× RTX A6000 (48 GB), 64 cores, ~126 GB free RAM.
  The node is **shared with other users** and has no scheduler.
- **Approach B** (own harness from day one, Phase 0 as its first user), using the
  official UCR/UEA train/test splits so published HC2 results (tsml-eval /
  timeseriesclassification.com) can be used where splits match.

- **Credentials (user, 2026-11-20).** The user supplied a Hugging Face token (gated
  checkpoints such as TabPFN v2.5, possibly UniShape/NuTime) and a Prior Labs TabPFN API
  key, approving use of the hosted TabPFN service for public UCR/UEA features. Both live
  only in `~/.config/tune-or-rocket/secrets.env` (mode 600, outside the repo) as
  `HF_TOKEN` and `TABPFN_TOKEN`; code reads them from the environment and never logs,
  prints or commits them. A gated download that still fails (licence not accepted on
  the account) is reported "not run (gated licence)" and raised with the user. Runs that
  use the hosted service are marked `hosted=true` in the run log; their timings are not
  comparable to local compute and are reported separately.

## 3. Goals, non-goals, anti-goals

Goals
- Phase 0 report (Section 7) answering the three facts above.
- A harness with: data loading on official splits, seeded support draws, a common FM
  adapter interface, one training loop, training-free arms, a cost logger, a GPU picker
  that respects the sharing cap, and an append-only run log.

Non-goals
- No accuracy, F1 or loss on any **test** split. No validation-accuracy comparisons
  between arms either: Phase 0 produces no score that could shape hypotheses. (Training
  loss may be logged to show a code path trains; it is not reported as a result.)
- No preregistration, no pilot, no dataset selection for the main study.
- No MONSTER, PTB-XL, Sleep-EDF or wearable pipelines yet (they come with Phase 1+).
- No HC2 runs.

Anti-goals (failure even if the code works)
- Touching a test split in Phase 0.
- Reporting a number that was not computed, or an estimated one as measured.
- Silently substituting an arm (e.g. LoRA for full) without reporting "not run (reason)".
- Occupying more than the sharing cap, or a GPU someone else is using.
- Timings measured under co-tenancy reported without a flag.

## 4. Constraints

- **Sharing cap:** at most 2 GPUs and 16 CPU cores at once. A job only starts on a GPU
  with < 1 GB used and 0% utilisation at launch; otherwise it waits.
- **Disk:** ~63 GB free on `/` (93% full). Phase 0 checkpoints (~15–20 GB) fit; datasets
  for later phases will not. Model and dataset caches go in one gitignored directory
  (`cache/`, with `HF_HOME` pointed there) so the footprint is visible and removable.
  Flag to the user before Phase 1 that more disk is needed.
- **Integrity (brief Section 11):** public data and checkpoints only; never invent a
  number; missing = reported missing; user's personal details never sent externally.
- **Benchmark exposure:** record, per model, which corpora the model paper says it was
  pretrained on (UCR/UEA exposure matters later).

## 5. Approaches considered

- **A. One-off probe scripts, harness later.** Fastest report; probe code discarded;
  integration problems (length/channel handling, heads, LoRA targets) surface later.
- **B. Own harness, Phase 0 as first user (chosen).** 2–3 days more before the report;
  measured costs are those of the real pipeline; Phase 1 starts directly.
- **C. Build on tsml-eval.** Training-free arms and HC2 results for free, but it is
  built around CPU sklearn-style estimators and fights GPU fine-tuning, LoRA, cost
  logging and few-label support draws.

Chosen: B, borrowing C's split convention.

## 6. Harness design

Python package `tor/` ("tune-or-rocket"). Python 3.11, PyTorch, `peft` for LoRA,
`aeon` for data and ROCKET-family transforms, `tabpfn`. Environment pinned in a lock
file; package versions written into every run record.

### 6.1 Data (`tor/data`)
- Load UCR and UEA via `aeon` on their official train/test splits. Phase 0 only needs
  metadata for all datasets and the full data of one or two small datasets.
- Metadata table (name, archive, n_train, n_test, length, channels, classes, equal
  length, missing values) taken from the **published dataset summaries**
  (timeseriesclassification.com / aeon's dataset lists), not computed from test files.
  Length percentiles for the probe shapes come from this published metadata. *(Builder
  call: keeps the test guard absolute.)*
- Test arrays are loaded only through an explicit `load_test()` that Phase 0 code never
  calls; a guard raises if it is called while `TOR_PHASE=0`. No metadata exemption.

### 6.2 Support draws (`tor/splits`)
- Seeded draws from the **training split only**: n per class (5, 20) or fraction (10%,
  100%), draws 0–9 by default. Saved as index files under `splits/` and committed, so
  every arm uses identical draws. Phase 0 implements and tests this; it uses draws only
  for the end-to-end smoke run.

### 6.3 FM adapter interface (`tor/models`)
One adapter per model family, all exposing:
- `load(init: "pretrained" | ("random", seed))` — random uses the package default
  initialisation.
- `preprocess(X)` — the model's own length/channel handling (resize, pad, patching,
  per-channel processing), as its paper or code does it; documented per adapter.
- `embed(X) -> features` — frozen embedding. Which part of the network is used and how
  it is pooled follows the model's own classification/embedding path where one exists
  (e.g. MOMENT's classification head input); otherwise the encoder output, mean-pooled
  over tokens (as Jain et al. do for Chronos-Bolt). Documented per adapter. Matching
  paper 1's extraction is not required (standalone paper). *(Builder call.)*
- `classifier(n_classes, mode)` with `mode ∈ {head, lora, full}` — returns a module
  with the right parameters trainable. LoRA rank and target modules come from the
  model's own recipe if one exists; otherwise a fixed default (rank 8, attention
  projections) recorded in the adapter.
- `default_recipe()` — the model paper's fine-tuning recipe (optimiser, lr, epochs,
  batch size) where published; source cited in the adapter docstring.

Candidates (all need verification): UniShape, Mantis-8M, MantisV2, NuTime, MOMENT-1
small/base/large, Chronos-Bolt small/base, Chronos-2, TimesFM-2.5, Toto, Reverso, plus
InceptionTime (or H-InceptionTime) as the non-FM deep reference. A model whose weights
or code are not public gets a row "not available (reason)" and no adapter.

### 6.4 Training loop (`tor/train`)
One loop for every FM mode and for from-scratch training: AdamW, fixed epochs or early
stopping on an inner split of the training data (both supported; Phase 0 uses fixed
epochs), mixed precision where the model supports it, gradient checkpointing as an
option. Records divergence (NaN/inf loss) as an outcome.

### 6.5 Training-free arms (`tor/baselines`)
MiniRocket + ridge, MultiRocket+Hydra + ridge, QUANT + extra trees (aeon), and
ROCKET-family features + TabPFN. Phase 0 runs them only for timing and to establish
TabPFN limits (max classes, samples, features; version; whether it needs the GPU).
Phase 0 checks **both** recipes, RocketPFN (ROCKET + TabPFN v2.5) and MASHT
(MultiRocket+Hydra + TabPFN-3), including each one's feature-reduction or ensembling
step, since TabPFN takes far fewer features than the transforms produce. Which one is
the arm is the user's decision after Phase 0 (brief §12.4). *(Builder call for Phase 0
scope.)*

### 6.6 Cost logger (`tor/cost`)
Wraps every run and records: wall-clock train / inference time, inference time per
1,000 series, `torch.cuda.max_memory_allocated`, GPU model, CPU cores used, and a
co-tenancy snapshot (nvidia-smi of the used GPU and node load average at start and
end). A run with any other process on its GPU, or node load above a set threshold, is
marked `shared=true`; shared timings are re-run on an idle card when possible and are
never reported unflagged.

### 6.7 GPU picker and run log
- **One GPU per run.** No data-parallel or sharded runs; the 2-GPU cap means two
  concurrent single-GPU runs. All candidates are well under 1B parameters, so a 48 GB
  card is expected to suffice; anything that does not is reported "not run (memory)".
  *(Builder call.)*
- `tor/gpu`: picks an idle GPU per Section 4, holds a lock file so at most 2 of our jobs
  run at once, exits cleanly if none is free within a timeout.
- `tor/runlog`: append-only JSONL, one record per run, keyed by a hash of the config;
  includes git commit, package versions, cost record, status (`ok`, `oom`,
  `diverged`, `not_run: reason`).

## 7. Phase 0 work and report

1. **Literature.** Re-run the search (arXiv, Semantic Scholar, OpenReview, 2026
   workshops) for fine-tuned TSFM vs ROCKET-family classification comparisons; read in
   full the three unread papers in brief 5.3; report overlaps with brief Q1–Q6 from full
   text, with links.
2. **Model availability.** For each candidate: weights URL, licence, parameter count
   (counted from the loaded model, not quoted), code path used, max context length,
   multivariate handling, documented pretraining corpora.
3. **Memory and time probes.** For each available model × mode {head, lora, full,
   scratch}: peak GPU memory and seconds per epoch on **synthetic tensors** of shapes
   taken from UCR/UEA metadata (series lengths at the 10th, 50th and 90th percentile and
   max of the archive; 1 and a representative multi-channel count; batch sizes 8, 32,
   64). Synthetic data means memory and time are measured without touching any real
   labels. Then one **end-to-end smoke run** per model and mode on the training split
   of one small UCR dataset (chosen by a seeded rule from metadata, recorded and
   declared in the report so it can be excluded or declared later) to prove the code
   path trains; no evaluation.
4. **Training-free timing and TabPFN limits** on the same shapes; TabPFN version and
   hard limits; which UCR/UEA datasets exceed them (from metadata).
5. **HC2 availability.** Whether published HC2 (and MR-Hydra) per-dataset results on
   the official splits exist for the UCR/UEA datasets we would use; note that they are
   full-label results only, so HC2 is a reference at the 100% budget.

**"Fits"** means: the mode runs one GPU at **batch 8, 90th-percentile UCR/UEA length,
single channel**, with gradient checkpointing allowed but recorded. The full grid is
reported alongside, so the steering side can see the margin. *(Builder call.)*

Report: `reports/phase0.md`, committed, in the brief's format (what was run, numbers,
deviations, anything unexpected, what is next). One table per model: available /
licence / parameters / fits full on 48 GB / fits LoRA / seconds per epoch (with shared
flag). A missing measurement is written as missing with its reason.

## 8. Left to the builder

- Package layout details, test framework, config format.
- Per-model preprocessing details, after reading each model's code.
- The node-load threshold for the shared flag, and the wait timeout of the GPU picker.
- Exact epochs for timing probes (enough to get a stable per-epoch time, e.g. 3 after
  1 warm-up).
- Parallelising the literature search with subagents.
