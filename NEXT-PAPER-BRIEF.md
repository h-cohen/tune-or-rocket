# Brief for paper 2: when is fine-tuning a time-series foundation model worth it for classification?

Version 1, 2026-11-19. Written by the steering agent for a new implementation agent
that has no other context. Read all of it before writing code. Where this brief says
"verify", the fact has not been checked against a primary source and must be checked
before it is used.

Authors (planned): Hadar Cohen and Prof. Raja Giryes, Tel Aviv University.

---

## 0. How to use this document

- **Roles.** You are the implementation agent: you plan runs, write code, run experiments
  and report results. A separate steering agent writes the paper and checks your
  reports. The user (Hadar Cohen) makes all decisions listed in Section 12 and passes
  messages between the two agents.
- **Before anything else**, do Phase 0 (Section 9): re-check the literature, confirm
  which checkpoints are public and runnable, and measure what fits on our GPUs. Report
  back before you build the full pipeline.
- **Nothing in this brief is a result of paper 2.** All numbers below come from paper 1
  (our finished study of frozen encoders) or from published papers. Do not copy them
  into paper 2 tables as if they were new measurements.
- **Integrity rules** (Section 11) are binding. The most important: public data and public
  checkpoints only; never invent or estimate a number; preregister before scoring;
  never change a preregistered verdict; label everything added afterwards as
  exploratory.

---

## 1. Summary in one paragraph

Paper 1 showed that frozen time-series foundation models (FMs) add nothing measurable to
MiniRocket features on the UCR archive at few labels, while ROCKET-family features add
several points to every FM. The obvious objection is that nobody uses FMs frozen: in
practice they are fine-tuned. Paper 2 answers that objection. It compares fine-tuned FMs
(full fine-tuning and parameter-efficient fine-tuning) with the strongest training-free
pipelines (MultiRocket+Hydra, ROCKET features with a tabular in-context classifier, and
HIVE-COTE 2 as an accuracy reference) on several benchmarks, across label budgets from a
handful per class to the full training set. Every FM is also trained from scratch with
the same recipe, so the gain from pretraining is isolated, and every arm's compute is
measured, so accuracy is reported against cost. The expected product is a map: in which
regimes (labels, data size, domain, series length, channels) a fine-tuned FM beats the
best training-free pipeline, by how much, at what cost, and how much of that is due to
pretraining. The paper is publishable whatever the map shows; the headline is decided by
the data, not in advance.

---

## 2. Where this comes from: paper 1 in brief

Paper 1 title (draft): "What Does Pretraining Add Beyond Random Kernels? A Controlled
Audit of Time-Series Foundation Models for Classification", target TMLR. Its files are in
`paper/` (LaTeX in `paper/sections/`, reports in `paper/data/`), with the original brief
in `PAPER-BRIEF.md` and the protocol in `FINAL-EXECUTION-PLAN.md`. The implementation
repository of paper 1 is on the remote machine at `/mnt5/home/hadar/jepa-rocket/audit/`
(not visible from the steering side); reuse its code where possible (Section 8).

### 2.1 Question and design of paper 1

- Question: if a classifier already uses MiniRocket features, does adding a frozen
  pretrained embedding improve it, and is any improvement due to pretraining?
- Three probes per dataset: MiniRocket alone (R), MiniRocket + pretrained embedding
  (R⊕E_pre), MiniRocket + embedding of a random-weight copy of the same network
  (R⊕E_rand, the "twin"; five twins per encoder, seeds 0, 7, 42, 123, 2026).
- Quantities: B(X) = acc(R⊕X) − acc(R) (what a block adds); G_alone = acc(E_pre) −
  acc(E_rand); G_cond = B(E_pre) − B(E_rand) (preregistered primary quantity).
- Probe: kernel ridge on the two blocks with a learned block weight w ∈ {0, .1, .25, .5,
  .75, .9, 1}, chosen by leave-one-out on the support set (plain concatenation at few
  labels costs several points by itself).
- Controls: Hydra as preregistered positive control; planted blocks of known strength;
  minimum detectable effect 0.39–0.53 points.
- Statistics: dataset as unit; paired sign-flip test with Holm correction; bootstrap
  intervals; ±1-point equivalence margin; amendment A17.2 (BCa bootstrap must agree with
  the sign-flip test, otherwise "inconclusive (test-sensitive)").
- Primary setting: 96 eligible UCR datasets, n = 5 labels per class, linear probe.
  Later preregistered addenda: UEA (20 datasets), three recent forecasters, three new
  wearable datasets. Pending at the time of writing: PTB-XL / Sleep-EDF (A3), MONSTER
  (A4).

### 2.2 Results of paper 1 that motivate paper 2

All from the paper 1 draft; UCR, n = 5, linear probe unless stated.

- Encoders tested: MOMENT-1-large, Mantis-8M, Mantis+, MantisV2, Chronos-T5-large; later
  Chronos-2, TimesFM-2.5, Toto.
- Alone, every encoder beats its twins by +1.48 to +22.93 points.
- Added to MiniRocket, a pretrained embedding changes accuracy by −0.34 to +0.41 points;
  every interval includes zero. Exploratory equivalence test: every original encoder is
  within ±1 point of zero (Holm p ≤ 0.015).
- Hydra adds +1.39 [0.74, 2.19]; MR-Hydra alone is 0.6 to 1.3 points more accurate than
  MiniRocket fused with any of the first five encoders (preregistered).
- MiniRocket adds 4.1 to 6.8 points to each encoder; each encoder alone is 3.9 to 7.1
  points below MiniRocket alone. With all labels, each encoder alone is still 3.6 to 5.0
  points below MiniRocket.
- TimesFM-2.5 and Toto lower MiniRocket's accuracy (−0.67, −0.63, intervals below zero).
  Chronos-2 meets the preregistered "adds" criterion only because its twin lowers
  accuracy (B(E_pre) = −0.01).
- Exception: wearable activity data. The Mantis encoders add several macro-F1 points on
  6 of 7 wearable datasets; preregistered hypothesis W1 confirmed on Capture-24 and HHAR,
  not on PAMAP2; the gain does not depend on unseen test subjects.
- Methodological lessons that carry over to paper 2:
  1. A random-weight control can mislead: if the random copy hurts, "pretrained minus
     random" is positive without any real gain. Always report each arm against the
     baseline, not only against each other.
  2. Fixed-weight concatenation and compressed baselines create apparent gains.
  3. A null result needs a positive control and a stated minimum detectable effect.
- Cost observed in paper 1: computing all UCR embeddings took about 37 GPU-hours on three
  11 GB GPUs; one-off timing gave about 4.6 ms per series for MOMENT-1-large and 9 to 42
  ms per series for Chronos-T5-large; loading a model takes 13 to 45 s. Timing for
  Mantis, MiniRocket and Hydra is still a TODO in paper 1.

### 2.3 The objection paper 1 cannot answer

Paper 1 is limited to frozen encoders with a linear probe. Its scope paragraph says
fine-tuned models may behave differently. Paper 2 tests exactly that.

---

## 3. The research question in depth

### 3.1 Main question

For time-series classification, when does adapting a pretrained foundation model to the
task (fine-tuning) give a more accurate classifier than the best training-free pipeline
built from random convolutional features, how large is the difference, how much of it is
due to pretraining rather than the architecture, and what does it cost?

### 3.2 Why it matters

- Practitioners choosing a method: training-free pipelines run on a CPU in seconds to
  minutes; fine-tuning an FM needs a GPU, a large checkpoint, hyperparameter choices and
  far more compute. The choice should rest on measured accuracy and cost per regime, not
  on the label "foundation model".
- FM developers: the claim that pretraining learns general time-series features should
  show up as a gain over the same architecture trained from scratch and over strong
  training-free features, at a reasonable cost.
- Evaluation practice: current comparisons mix protocols (frozen vs fine-tuned, weak vs
  strong baselines, with or without controls, rarely with cost). A controlled protocol
  is itself a contribution.

### 3.3 Sub-questions

- **Q1 (accuracy gap).** At each label budget, is the best fine-tuned FM more accurate
  than the best training-free pipeline? By how much, with what uncertainty?
- **Q2 (pretraining).** Is a fine-tuned FM more accurate than the same architecture
  trained from scratch with the same recipe and tuning budget? (This is the fine-tuning
  analogue of paper 1's random-weight twin.)
- **Q3 (adaptation).** How much does fine-tuning add over the frozen embedding (linear
  probe) and over frozen embedding fused with MiniRocket (paper 1's probe)? Full vs
  parameter-efficient (LoRA) vs head-only.
- **Q4 (cost).** What is the compute cost of each arm (training, tuning, inference), and
  where are the arms on the accuracy–cost Pareto front? At what accuracy gain per unit
  of compute does an FM become the better choice?
- **Q5 (where).** Which properties of a dataset predict an FM win: number of labels,
  training-set size, series length, number of channels, number of classes, domain
  (human motion, medical, audio, sensors, images-as-series), distribution shift between
  fit and test (new subjects, new devices)?
- **Q6 (break-even).** For each FM and dataset family, the label budget at which the
  fine-tuned FM matches the best training-free pipeline (a break-even point, in the
  sense of the forecasting break-even analysis of arXiv 2607.04919).

### 3.4 Possible outcomes and the paper each gives

| Outcome | Headline | Publishable? |
|---|---|---|
| Fine-tuned FMs never beat the best training-free pipeline beyond the margin, at any budget | "Use ROCKET-family pipelines; fine-tuning does not pay for classification on these benchmarks" | Yes, strong practical message; must survive the R2R counter-evidence (Section 5) |
| FMs win only with many labels and only the best classification FM, by a small margin at large cost | "Fine-tuning pays only in a narrow regime; here is the map and the cost" | Yes, most likely outcome given current evidence |
| FMs win broadly once fine-tuned | "Frozen evaluations understate FMs; fine-tuning closes and reverses the gap" | Yes, corrects paper 1's practical reading |
| FMs win only in specific domains (e.g. wearables, medical) | "Domain decides; a decision rule for practitioners" | Yes |
| Pretrained ≈ from scratch after fine-tuning | "The architecture, not pretraining, carries the gain" | Yes, and important for FM developers |

Do not choose the headline before the data. The steering agent and the user will.

---

## 4. Background the implementation needs

### 4.1 ROCKET family (training-free)

- ROCKET (Dempster et al. 2020): convolves each series with thousands of random kernels,
  pools each output (max and proportion of positive values), linear classifier.
- MiniRocket (2021): small, almost deterministic kernel set, one pooling operator (PPV),
  about 10,000 features, much faster at similar accuracy.
- MultiRocket (Tan et al. 2022): adds pooling operators and the first-difference series.
- Hydra (Dempster et al. 2023): groups of competing random kernels, counts which kernel
  wins. MultiRocket+Hydra ("MR-Hydra", "MR-H") is one of the strongest non-ensemble
  classifiers on UCR.
- QUANT (Dempster et al. 2024): quantiles over dyadic intervals, extra-trees classifier.
- ROCKET features + TabPFN: RocketPFN (arXiv 2606.21786) and MASHT (MultiRocket+Hydra +
  TabPFN-3, arXiv 2607.19234) match HIVE-COTE 2 without task training (details in
  Section 5).
- HIVE-COTE 2 (HC2): the accuracy reference on UCR; very expensive (one benchmark reports
  about 694 CPU core-hours on 140 cells against 2.8 for MultiRocket; see AIMultiple in
  Section 5). Use published HC2 results where splits match, or run on a subset only.
- Implementations: `aeon` (used in paper 1 for MiniRocket, MultiRocket, Hydra, QUANT).

### 4.2 Foundation models and adaptation modes

- Frozen embedding + linear probe: paper 1's setting.
- Head-only: train a classification head on frozen features (close to the linear probe).
- Parameter-efficient fine-tuning: LoRA adapters (rank, target modules to be fixed), or
  model-specific adapters (e.g. the channel-reduction adapters of arXiv 2409.12264).
- Full fine-tuning: all weights trained with a classification head.
- From scratch: the same architecture and head, randomly initialised (package default
  initialisation, as for paper 1's twins), trained with the same recipe and tuning budget.
  This is the control for Q2.
- In-context / zero-shot classification FMs (e.g. TiCT, arXiv 2511.19694; the zero-shot
  in-context approach of arXiv 2602.00620): optional extra arm; they need no
  fine-tuning and belong to the "no task training" side of the cost axis.

### 4.3 Candidate models

Selection criteria: public weights under a licence that allows research use; a supported
code path for classification fine-tuning; fits our GPUs at least with LoRA; covers both
families (classification-pretrained and forecasting-pretrained). Every row needs Phase 0
verification of availability, size, licence and a working fine-tuning path.

| Model | Family | Size | Why include | Status |
|---|---|---|---|---|
| UniShape | classification FM | verify | Best fine-tuned TSFM in R2R (beats MR-Hydra on average rank) | verify public weights and code |
| Mantis-8M, MantisV2 | classification FM | 8M (Mantis) | Paper 1 encoder; strongest frozen FM on UEA in Isiacik & Ifrim; wearable exception | public (paper 1) |
| NuTime | classification FM | verify | In R2R and Isiacik & Ifrim | verify |
| MOMENT-1 (small/base/large) | general FM | large reported as about 385M | Paper 1 encoder; adapters paper fine-tunes it | public (paper 1); large likely needs LoRA |
| Chronos-Bolt (small/base) | forecaster | base 205M (Jain et al.) | Forecaster family; Jain et al. used it frozen | verify fine-tuning path |
| Chronos-2, TimesFM-2.5, Toto | forecasters | verify | Paper 1 recent encoders | public (paper 1); fine-tuning path to verify |
| Reverso | forecaster | 2.6M (Jain et al.) | Direct replication point with Jain et al.'s fine-tuning arm | verify |

Plus at least one supervised deep baseline trained from scratch that is not an FM
architecture (InceptionTime or H-InceptionTime), as the standard deep-learning
reference.

---

## 5. Related work (state of 2026-11-19)

Checked against full text by fetch agents on 2026-11-19 unless marked "unverified".
Re-run the search at the start (Phase 0) and before submission; this area moved fast
in October–November 2026.

### 5.1 Closest work

**Jain, Kar, Bhadane, Patil, "Big Models, Small Gains: Frozen Forecasters as Classifiers
on MONSTER", arXiv 2611.06988, 4 Nov 2026, ACML 2026 FMLTS workshop.**
- Models: Reverso (2.6M parameters) and Chronos-Bolt base (205M), frozen; Reverso also
  fine-tuned end to end.
- Baselines: MiniRocket, MultiRocket, Hydra (ridge), QUANT (extra trees). They do not
  use TabPFN or RocketPFN; RocketPFN is cited in related work only.
- Data: six MONSTER datasets (UCIActivity, WISDM2, CrowdSourced, InsectSound,
  FordChallenge, LenDB; 10,299 to 626,180 series), predefined folds, label budgets 1%,
  10%, 100%. One random-weight copy per model (seed 0).
- Frozen result: frozen representations are worse than the best non-pretrained baseline
  in 15 of 17 settings and tie in 2; pretraining beats random weights on 5 of 6 datasets,
  by up to 23 points.
- Fine-tuning arm (Table 3): Reverso only, on UCIActivity, CrowdSourced and InsectSound,
  1% and 10% labels, folds 0–1. No validation split; the last epoch is evaluated.
  Fine-tuned Reverso beats the same network from random weights in 17 of 18 runs at 1%,
  but "stays below the best baseline in every case, by 1.5 points at the closest
  (InsectSound, 1%)", at "about ten times the compute". Section heading: "Fine-tuning
  does not close the gap."

  | Dataset | Labels | Fine-tuned | From scratch | Linear probe | Best baseline |
  |---|---|---|---|---|---|
  | UCIActivity | 1% | 66.9 | 59.2 | 61.3 | MiniRocket 93.8 |
  | UCIActivity | 10% | 84.5 | 80.0 | 77.9 | MiniRocket 97.0 |
  | CrowdSourced | 1% | 64.4 | 61.2 | 65.6 | MultiRocket 67.0 |
  | CrowdSourced | 10% | 68.9 | 69.9 | 69.4 | Hydra 71.9 |
  | InsectSound | 1% | 62.0 | 59.0 | 42.8 | QUANT 63.4 |
  | InsectSound | 10% | 71.7 | 68.0 | 56.8 | Hydra 75.6 |

- Timing (their Table 9, seconds per fold, "indicative only"): MiniRocket 2–8 s,
  MultiRocket 4–26 s, Hydra 13–356 s, QUANT 5–26 s, Reverso embeddings 193–656 s,
  Chronos-Bolt embeddings 31–114 s, one Reverso fine-tuning run at 10% labels
  2,324–5,824 s.
- Their stated limitation: "fine-tuning experiments on three datasets with two folds".
  They do not announce fine-tuning of larger models as future work.
- What it leaves open: one small forecaster, three datasets, no full-label fine-tuning,
  no classification FMs, no LoRA, no TabPFN-based baseline, no tuning protocol, one seed.

**Liu et al., "From Retrieval to Recognition: Knowledge-Enhanced Post-training of Time
Series Foundation Models for Classification" (R2R), arXiv 2610.26114, 19 Oct 2026.**
- Full supervised fine-tuning of NuTime, Mantis and UniShape on 128 UCR datasets, with
  and without their retrieval method R2R; zero-shot test on 30 more datasets; UEA in the
  appendix. Hardware (from a summary, unverified): two RTX A6000; up to 500 epochs.
- Main table, average rank over 128 UCR datasets (lower is better): R2R_UniShape 3.01,
  fine-tuned UniShape 4.47, HC2 5.16, R2R_Mantis 6.18, MR-Hydra 6.27, RDST 7.38,
  Rocket 7.62, MiniRocket 7.75, R2R_NuTime 7.90, fine-tuned Mantis 8.07, fine-tuned
  NuTime 10.26 (others: InceptionTime 9.41, TS2Vec 12.65, PatchTST 18.19).
- So: fine-tuned UniShape beats MR-Hydra on average rank at full labels (not HC2);
  fine-tuned Mantis and NuTime do not.
- No from-scratch control, no GPU-hours. Their ablation table gives different ranks for
  the same methods (R2R_UniShape 1.48) on an unexplained basis. Mean accuracies were not
  read.
- Implication for paper 2: a blanket "ROCKET is better" claim is already contradicted at
  full labels on UCR. UniShape must be in the study; replicating its gain over MR-Hydra
  under a controlled protocol, with cost and a from-scratch control, is itself a result.

**Isiacik and Ifrim, "Random Initialization as a Baseline for Time Series Foundation
Models", arXiv 2610.30950, 22 Oct 2026 (NeurIPS 2026 TS-LIMITS workshop per
`PAPER-BRIEF.md`).**
- Seven frozen model instances (MOMENT small/base/large, Mantis, MantisV2, UniTS,
  NuTime), five random seeds each {0, 7, 42, 123, 2026}, four downstream classifiers, 30
  UEA datasets.
- Pretrained vs random: +2.58 points pooled (1.43 to 5.19 per instance); only NuTime
  significant after Holm.
- On 24 common datasets: ROCKET 0.754, Hydra 0.752, QUANT 0.752, best FM Mantis 0.731.
- Frozen only; fine-tuning is explicitly out of scope.

### 5.2 Training-free pipelines with tabular in-context models

- **RocketPFN**, O'Rourke, Trišović, Bertsimas, arXiv 2606.21786 (June 2026): ROCKET
  features + TabPFN v2.5; not significantly different from HC2 (paper 1's check: "matches
  HC2", ≤ 10 classes only); MOMENT, Mantis and MantisV2 embeddings significantly worse
  than ROCKET features under the same classifier.
- **MASHT**, Cüppers and Vreeken, arXiv 2607.19234 (July 2026): MultiRocket+Hydra
  features + TabPFN-3, no task training. UCR 112: mean accuracy MASHT 0.892, HC2 0.891,
  MR-Hydra 0.884 (MASHT beats MR-Hydra, ties HC2). "Multiverse" multivariate (71
  datasets): HC2 0.805, MR-Hydra 0.797, MASHT 0.795, no significant difference. Median
  runtime about 59.7 s per dataset on an A100. No TSFMs compared.

### 5.3 Fine-tuning and pretraining in time-series classification

- **PHIT**, Ismail-Fawaz et al., arXiv 2311.14534 (2024): pretrained H-InceptionTime
  (pretext task: which dataset a series comes from), fine-tuned per dataset on 88 UCR
  datasets; beats the same network trained from scratch (48 vs 23 wins). Comparison with
  MultiRocket / MR-Hydra / HC2 only from a summary: MR-Hydra reportedly still better
  overall (unverified).
- **Mantis**, Feofanov et al., arXiv 2502.15637 (v2, June 2026): full fine-tuning
  described in its appendix (all layers, 500 epochs, AdamW); no ROCKET-family baseline in
  the parts read.
- **Foundation model adapters**, Feofanov et al., arXiv 2409.12264 (2024): MOMENT
  (341M as stated there) and a ViT on 12 UEA datasets; full fine-tuning, head-only and
  channel-reduction adapters; up to 10x faster without accuracy loss; no ROCKET-family
  baseline, no from-scratch control.
- **AIMultiple benchmark** (Kalelioğlu, updated Aug 2026, not peer reviewed): 13 methods
  on 33 UCR/UEA datasets; FMs frozen with a logistic-regression probe. On 15 univariate
  datasets: RocketPFN 0.905, HC2 0.898, MultiRocket 0.890, Hydra 0.877, MiniRocket 0.876,
  MOMENT 0.853, Chronos-2 0.835, TiRex 0.826. CPU core-hours on 140 cells: HC2 694,
  MultiRocket 2.8, Hydra 2.1, MiniRocket 0.48; FM GPU time not recorded. "Every
  surviving contrast is a convolutional transform beating a pretrained model."
- **Auer et al.**, arXiv 2510.26777 (NeurIPS 2025 workshop): frozen forecasters as
  zero-shot feature extractors; no ROCKET.
- **TiCT** (arXiv 2511.19694) and the in-context approach of arXiv 2602.00620: zero-shot
  classification FMs; not fine-tuning; candidates for an optional arm.
- Not yet read (found by search; read in Phase 0): "An Empirical Evaluation of
  Foundation Models for Multivariate Time Series Classification" (2025; ResearchGate
  blocked, find another copy); "Improving Multivariate Time Series Classification with
  Class-Wise Training and Model Aggregation" (arXiv 2609.07493); "Is Self-Pretraining
  really useful to improve diagnosis in medical Time Series?" (arXiv 2608.06122).

### 5.4 Cost framing from forecasting

- **"When Do Foundation Models Pay Off? A Break-Even Analysis of Pretrained Time Series
  Forecasters"**, Tan Jerome and Simon, arXiv 2607.04919 (v2, 18 Nov 2026): forecasting
  only; Chronos-Bolt and Moirai zero-shot, LoRA (rank 8) and full fine-tuning against
  classical forecasters trained on 2% to 100% of the data; break-even point = smallest
  training size at which a classical method matches the best zero-shot FM. LoRA "rarely
  helps"; full fine-tuning beats zero-shot on 9 of 30 datasets. Use its break-even idea
  for Q6.

### 5.5 The gap

No study compares several fine-tuned TSFMs (classification-pretrained and forecasting-
pretrained) with the strongest training-free pipelines (MR-Hydra, ROCKET+TabPFN, HC2 as
reference) with all of: a from-scratch control for every FM, several label budgets from
few to all, a fixed and equal tuning budget, several benchmarks including domain-shift
settings, and measured compute. Jain et al. did a small version (one tiny forecaster,
three datasets, two budgets). R2R did one slice (full labels, UCR, no control, no cost)
and found that fine-tuned UniShape can beat MR-Hydra. Novelty wording for the paper:
"no controlled comparison of this kind exists", not "nobody fine-tuned FMs against
ROCKET".

---

## 6. Study design (draft, to be fixed in the preregistration)

### 6.1 Arms

Training-free (no gradient training of a network):
1. MiniRocket + ridge (`aeon` defaults; paper 1 settings).
2. MultiRocket + Hydra (MR-Hydra) + ridge.
3. QUANT + extra trees.
4. ROCKET-family features + TabPFN (RocketPFN and/or MASHT recipe; check class and size
   limits of TabPFN; record where it cannot run).
5. HC2 as accuracy reference: published results where the splits are identical,
   otherwise run on a subset; never let HC2 cost dominate the compute budget.

Foundation-model arms, for each FM:
6. Frozen embedding + linear probe (as paper 1; reuse embeddings where they exist).
7. Frozen embedding fused with MiniRocket under paper 1's learned-weight probe (reuse).
8. Head-only training on frozen features.
9. LoRA fine-tuning (fixed rank and target modules per model, from the model's own
   recipe if one exists).
10. Full fine-tuning (where it fits in memory; otherwise state "not run" with the
    reason).
11. From scratch: same architecture and head, random initialisation, same recipe and
    same tuning budget as arm 10 (or 9 if full does not fit).

Deep supervised reference:
12. InceptionTime or H-InceptionTime from scratch (standard recipe).

Optional: in-context classification FMs (TiCT or similar) as a zero-training FM arm.

### 6.2 Datasets

Choose and freeze the list before any score exists (eligibility rules from metadata
only, as in paper 1).

- UCR archive: 128 datasets; paper 1 used 96 eligible (equal length, no missing values,
  length ≥ 16, ≥ 5 training series per class). The common literature subset is 112
  (equal length). Decide which; report the list.
- UEA archive: 30 datasets; paper 1 used 20 (eligibility plus a size limit). Isiacik &
  Ifrim and MASHT use other subsets; record overlap.
- MONSTER (large datasets; Jain et al.'s six datasets at minimum, for direct comparison).
  Paper 1's A4 run builds this pipeline; reuse it.
- Wearable activity data: paper 1's seven datasets (UCI HAR, MHEALTH, HARTH, WISDM,
  Capture-24, HHAR, PAMAP2), subject-disjoint folds, macro-F1. This is where frozen
  Mantis already helps, so it is the natural test of whether fine-tuning widens the gap.
- Medical: PTB-XL, Sleep-EDF (paper 1's A3 pipeline), subject or patient splits.
- Domain shift: subject-split vs window-split (paper 1's Q-W2 design), device shift
  where a dataset allows it (e.g. HHAR devices; verify).
- Public data only.

### 6.3 Label budgets

At least: n = 5 per class (paper 1's primary), n = 20 per class, 10% of the training
set, and 100%. Few-label fine-tuning needs a model-selection rule that does not use the
test set; see 6.4. Use the same support draws for every arm (paper 1 used ten draws).
Large datasets (MONSTER) may use fewer draws; fix this in advance.

### 6.4 Tuning protocol (fairness)

- Every arm gets the same tuning budget, stated in trials or GPU/CPU time, decided in the
  preregistration. Default recipes from each model's paper are the first trial.
- Model selection only on data from the training split: inner k-fold or a held-out part
  of the training set; at very few labels, leave-one-out or a fixed recipe with no
  selection (state which). Jain et al. used no validation split and took the last epoch;
  either replicate that as one arm or justify the change.
- Fixed number of epochs or early stopping on the inner validation split; record
  divergence and instability (the break-even paper found full fine-tuning unstable).
- Seeds: at least three training seeds per FM configuration (paper 1 used five twins);
  report seed variance. MiniRocket's own random state also moves accuracy (Jain et al.:
  59.1 to 65.5% on one setting); fix and report it.
- The test split is touched once, by the final evaluation.

### 6.5 Cost measurement

For every arm and dataset: wall-clock training time, tuning time, inference time per
1,000 series, peak GPU memory, hardware used (GPU model, CPU cores). Report GPU-hours and
CPU core-hours separately. Pretraining cost of the FMs is reported as a known external
cost where the model paper states it, never estimated. Times measured while other jobs
share the machine must be flagged (Jain et al. had this problem).

### 6.6 Quantities

For dataset d, budget b:
- Δ_TF(FM) = acc(FM, fine-tuned) − acc(best training-free arm), where "best" is fixed in
  advance (MR-Hydra as primary; ROCKET+TabPFN as secondary).
- Δ_pre(FM) = acc(FM, fine-tuned) − acc(same architecture, from scratch).
- Δ_adapt(FM) = acc(FM, fine-tuned) − acc(FM, frozen + linear probe).
- Cost ratio = compute(FM arm) / compute(best training-free arm).
- Break-even budget (Q6): smallest budget at which Δ_TF is not significantly negative.

Report each arm against the training-free baseline, not only arms against each other
(lesson 1 of paper 1).

### 6.7 Statistics

- Dataset (or subject fold) as unit; paired differences across datasets per FM and
  budget.
- Paired sign-flip or Wilcoxon test with Holm correction within each preregistered
  family; bootstrap intervals (and paper 1's BCa agreement rule); equivalence margin
  (paper 1 used ±1 point; decide).
- Positive control: an arm known to beat MiniRocket (MR-Hydra) must be detected; state
  the minimum detectable effect from interval half-widths.
- Secondary: average ranks and critical-difference diagrams (comparability with R2R and
  MASHT), Pareto plots of accuracy against cost.
- Q5 (dataset properties): preregister the properties and the correlation tests; Holm.

### 6.8 Draft hypotheses (to be decided by the user before preregistration)

- H1: At n ≤ 20 labels per class, no fine-tuned FM is more accurate than MR-Hydra on
  average (UCR), and the difference is within the equivalence margin or negative.
- H2: At 100% labels, at least one classification-pretrained FM (UniShape) is more
  accurate than MR-Hydra on UCR (replication of R2R under a controlled protocol).
- H3: For every FM, fine-tuning from pretrained weights is more accurate than training
  the same architecture from scratch, with the gap largest at few labels.
- H4: On wearable data, fine-tuned Mantis encoders beat MR-Hydra at every budget.
- H5: The best FM arm costs at least two orders of magnitude more compute than the best
  training-free arm on every benchmark (descriptive unless a test is defined).

---

## 7. Compute

- Available: three GPUs with 11 GB each (paper 1). Mantis-8M, Reverso and small/base
  variants should fine-tune fully; MOMENT-large and Chronos-T5-large likely need LoRA,
  gradient checkpointing and small batches. Measure in Phase 0, do not assume.
- Scale: models × datasets × budgets × seeds × arms multiplies quickly. Phase 1 must
  produce a measured cost per run so that the steering side can size the full study and
  the user can decide whether to request more compute.
- If a model cannot be fine-tuned in full on our hardware, report "not run (memory)"
  rather than silently switching the arm.

---

## 8. Reuse from paper 1

- Code: data loaders, eligibility rules, support-draw generation, MiniRocket/Hydra/QUANT
  pipelines, frozen embeddings and twins, statistics (sign-flip, Holm, bootstrap, BCa
  rule), reporting scripts, the deviation log format.
- Embeddings already computed for UCR, UEA and the wearable datasets (frozen arms 6 and
  7 need no new GPU time where they exist).
- The preregistration and amendment practice: commit the protocol before any score,
  log every deviation with date and reason, never open verdict files before the
  amendment that governs them is committed (paper 1 had to disclose an incident here).

---

## 9. Phases and deliverables

**Phase 0: checks (no scores).**
1. Re-run the literature search (arXiv, Semantic Scholar, OpenReview, NeurIPS/ICLR/ICML
   2026 workshops) for fine-tuned TSFM vs ROCKET comparisons; read the three unread
   papers in 5.3. Report anything that overlaps with 3.3 by full text, not abstract.
2. For each candidate model in 4.3: public weights, licence, size, fine-tuning code
   path, memory for full and LoRA fine-tuning at our sequence lengths, time per epoch on
   one UCR dataset.
3. Confirm the TabPFN version and limits for the ROCKET+TabPFN arm; confirm HC2 result
   availability for our splits.
Deliverable: a short report with a table per model (available / size / licence / fits
full / fits LoRA / seconds per epoch) and the literature findings.

**Phase 1: pilot (go/no-go).**
- Three FMs chosen to span size and family (proposal: Mantis-8M, UniShape, one mid-size
  forecaster such as Chronos-Bolt base or MOMENT base).
- About 20 UCR/UEA datasets plus two wearable datasets, chosen by a seeded rule before
  any score.
- Budgets n = 5 per class and 100%.
- Arms: MR-Hydra, ROCKET+TabPFN, frozen + linear probe, LoRA, full fine-tuning where it
  fits, from scratch.
- Pilot scores are pilot data: they may shape the protocol but are excluded from the
  confirmatory analysis, and their datasets are either excluded from the main study or
  declared in the preregistration.
Deliverable: accuracy per arm and dataset, measured cost per run, failures, and an
estimate of total compute for the full study.

**Phase 2: preregistration.** The steering agent drafts it from the pilot report; the
user approves; you commit it before any confirmatory score exists.

**Phase 3: full runs**, in the preregistered order. Report after each benchmark.

**Phase 4: analysis and reports**: per-benchmark tables, Pareto plots, break-even
analysis, deviation log, per-dataset CSVs.

Report format to the steering side (pasted by the user): what was run, the numbers with
intervals and test results, deviations, anything unexpected, and what is next. Never
round away a sign; never report a number you did not compute.

---

## 10. Risks

- **Scooping.** The area moved fast in Oct–Nov 2026; Jain et al. could extend their
  fine-tuning arm. Mitigation: move to the pilot soon after paper 1 is submitted;
  re-search monthly.
- **Unfair tuning.** FM papers tune hard; training-free baselines use defaults. Equal,
  stated budgets and default-first trials address this; report sensitivity to the budget.
- **Few-label fine-tuning instability.** Expect divergence at n = 5; record it as a
  result, not a bug to hide.
- **Compute.** The full grid may not fit; prioritise by the preregistered order and say
  what was not run.
- **Benchmark exposure.** Mantis-8M and MOMENT saw UCR/UEA training series in pretraining
  (paper 1, Appendix on exposure). Record exposure per model and dataset; include
  datasets no corpus is documented to contain.
- **Inconsistent published numbers.** R2R's two tables disagree; do not build a
  hypothesis on a single published number without reproducing it.

---

## 11. Integrity rules (binding)

- Public data and public checkpoints only.
- Never invent or estimate a number. A missing number is reported as missing.
- Preregister before any confirmatory score exists; never change a preregistered
  verdict; show amended verdicts next to the preregistered ones.
- Everything added after the preregistration is labelled exploratory and stays out of
  the confirmatory test families.
- Claims are limited to the models, data, budgets and protocol tested: "the FMs we
  tested", not "foundation models".
- Disclose all pilot runs and all deviations.
- The user's personal details are private; never send them to external services.

---

## 12. Decisions for the user

1. Go ahead with Phase 0 now, or after paper 1 is submitted.
2. Target venue (TMLR, or NeurIPS Datasets & Benchmarks with a released harness and
   leaderboard).
3. The model list after Phase 0, in particular whether UniShape and HC2 are required.
4. Equivalence margin and the primary training-free baseline (MR-Hydra vs ROCKET+TabPFN).
5. Hypotheses H1–H5: keep, change or drop.
6. Whether to request additional compute (decided after the Phase 1 cost estimate).
7. Whether to involve collaborators with FM fine-tuning experience.

---

## 13. Glossary

- **FM / TSFM**: time-series foundation model, pretrained on a large corpus.
- **Training-free pipeline**: random-feature transform (ROCKET family) plus a classifier
  that is fitted but not trained by gradient descent through a network (ridge, extra
  trees, or TabPFN in context).
- **MR-Hydra / MR-H**: MultiRocket and Hydra features concatenated, ridge classifier.
- **From scratch**: the FM architecture with random initial weights, trained on the task
  with the fine-tuning recipe; the paper 2 analogue of paper 1's twin.
- **Twin**: paper 1's frozen random-weight copy of an encoder.
- **Budget**: number of labelled training series used (per class or as a fraction).
- **Points**: accuracy or macro-F1 × 100.
- **Break-even budget**: the smallest label budget at which a fine-tuned FM is no longer
  significantly worse than the best training-free pipeline.

---

## 14. Sources

- Jain et al., arXiv 2611.06988: https://arxiv.org/html/2611.06988
- R2R, arXiv 2610.26114: https://arxiv.org/html/2610.26114
- Isiacik & Ifrim, arXiv 2610.30950: https://arxiv.org/html/2610.30950
- RocketPFN, arXiv 2606.21786: https://arxiv.org/html/2606.21786
- MASHT, arXiv 2607.19234: https://arxiv.org/html/2607.19234v1
- PHIT, arXiv 2311.14534: https://arxiv.org/html/2311.14534
- Mantis, arXiv 2502.15637: https://arxiv.org/html/2502.15637v2
- Foundation model adapters, arXiv 2409.12264: https://arxiv.org/html/2409.12264
- Break-even analysis, arXiv 2607.04919: https://arxiv.org/html/2607.04919
- Auer et al., arXiv 2510.26777: https://arxiv.org/abs/2510.26777
- TiCT, arXiv 2511.19694: https://arxiv.org/pdf/2511.19694
- Zero-shot in-context TSC, arXiv 2602.00620: https://arxiv.org/pdf/2602.00620
- AIMultiple benchmark: https://aimultiple.com/time-series-classification
- Paper 1 materials: `PAPER-BRIEF.md`, `FINAL-EXECUTION-PLAN.md`, `paper/` (draft,
  `paper/OPEN-QUESTIONS.md`, `paper/RELATED-WORK-VERIFICATION.md`).
