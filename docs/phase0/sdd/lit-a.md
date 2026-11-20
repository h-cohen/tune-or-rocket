# lit-a: literature search for paper 2 (searched 2026-11-20)

Scope: WebSearch (extended) x ~10 queries, WebFetch full-text HTML for each paper below. Not done: OpenReview TS-LIMITS group page (not opened; only the neurips.cc virtual page snippet via search), Semantic Scholar, Google Scholar (not reachable directly), ACML FMLTS page. Absence of hits is therefore not proof of absence.
Papers already in brief section 5 are not re-reported except where new facts were found (UniShape primary paper, Isiacik ECML).
Only items read in HTML full text are marked "full text". The search-tool answers were not used as evidence.

Bottom line: no new paper found that does a controlled fine-tuned-TSFM vs ROCKET-family comparison with from-scratch control, label budgets and cost. No high-threat item. Closest new items are medium/low.

## A. Fine-tuning evidence (new or newly detailed)

### A1. UniShape primary paper (brief only had it via R2R) - threat: MEDIUM
- Liu, Wang, Li, Zheng, Eldele, Wu, Ma. "A Unified Shape-Aware Foundation Model for Time Series Classification". arXiv 2601.06429, 10 Jan 2026. Full text (HTML).
- Models: UniShape (3.1M params), pretrained on ~1.89M univariate samples (UCR, UEA train splits + 8 more; so UCR train data are in pretraining). Foundation baselines: GPT4TS, MOMENT, UniTS, NuTime, Mantis; Chronos and Moirai zero-shot.
- Baselines: Rocket, MiniRocket, RDST, MultiRocket-Hydra (MR-H), InceptionTime, TS2Vec, PatchTST, TimesNet, SoftShape. HC2 not in the portion read.
- Data: 128 UCR fully supervised; 30 extra datasets zero-shot.
- Protocol: full fine-tuning, up to 300 epochs, CE + shape loss (mu=0.01), 5 seeds, checkpoint with lowest TRAIN loss reported. Four RTX A6000. Runtime analysis in Appendix F (not read), no figures.
- Key numbers: UniShape avg acc 0.8708, avg rank 2.71; MR-H 0.8621, rank 3.97; Wilcoxon p=2.96e-2 (about +0.9 pt).
- From-scratch control: yes, "w/o Pretraining" ablation on 18 UCR datasets only; pretraining wins; Transformer-encoder variant loses 1.64 pts without pretraining.
- Label budget: none (no few-shot). The labeling-ratio analysis varies the fraction of labels used in PRETRAINING (0/1/10/50/100%: acc 0.8395 -> 0.8588 on 18 UCR datasets), not downstream.
- Overlaps: Q1 (full labels only), Q2 (partial, 18 datasets, own model). Threat note: it already shows a fine-tuned pretrained model beating MR-H at full labels with a scratch ablation; paper 2 must cite it and not claim novelty on that point. Differences: no label curve, no cost, one model, UCR train in pretraining (leakage question for UCR).

### A2. Li, Liao, Chen, Zhao, "Data-Efficient Motor Condition Monitoring with Time Series Foundation Models", arXiv 2511.23177 (v2; Nov 2025, brief missed it) - threat: LOW-MEDIUM
- Full text (HTML) read.
- Models: MOMENT small/base/large, Mantis (8M). Full fine-tuning and LoRA.
- Baselines: CNN, LSTM, Transformer (= MOMENT architecture without pretraining, i.e. a from-scratch control), TSSequencerPlus, "MINIROCKET" (text says MiniRocket, cites MultiRocket paper), TSPerceiver, XCM, PatchTST; SVM/RF/DT/kNN on features.
- Data: ONE dataset, PMSM motor vibration/current, 45 classes, 4 channels x 512, ~960k samples. Training-data ratios 1%, 5% (primary), up to 80%.
- Stated results: LoRA ~ full fine-tuning at small sample sizes (large MOMENT+LoRA over 70% at 5%); Mantis full FT feasible on CPU. SVM on frozen features (Table III): 1%: raw 39.7, MOMENT 40.5, Mantis 84.1; 5%: 52.7 / 53.9 / 90.9; 80%: 52.7 / 54.4 / 91.3.
- Fine-tuned accuracies vs MiniRocket are only in figures (not readable from text). Abstract says Mantis reaches 90% at 1%; Table III gives 84.1% at 1% (SVM on frozen), inconsistency noted.
- Compute: no wall-clock, GPU-hours or FLOPs measured (qualitative only).
- Overlaps: Q2 (partial), Q3 (LoRA vs full), label curve on one industrial dataset. Threat low-medium: single domain, no cost, no ROCKET-family beyond MiniRocket, claims FM win; but a label-budget + scratch + LoRA design exists at small scale.

### A3. Coser, Orvieto, Soda, Zollo, "Is Self-Pretraining really useful to improve diagnosis in medical Time Series?", arXiv 2608.06122 (v2 7 Aug 2026) - threat: LOW
- Full text read. Small transformer encoders (1-3 layers), masked-reconstruction self-pretraining on the target data itself vs from-scratch. Datasets: CAMARGO 2021, Non-EEG Stress, Gait in Parkinson's (LOSO / subject-wise). A100.
- Numbers: +9.9 pts (Camargo Gyr-Y, 3 layers, 0.524 -> 0.623), multivariate Parkinson 0.895 -> 0.964/0.966; no ROCKET or external baselines, no cost, no label budgets.
- Overlaps: Q2 only (self-pretraining on target data, not a TSFM). Useful as related work on from-scratch controls.

## B. Training-free / in-context competitors (not fine-tuning)

### B1. TS2TabPFN, da Costa Merlin and Furtado Silva, arXiv 2608.04174 (Aug 2026) - threat: LOW
- Full text read. Features (catch22, tsfresh, MultiROCKET) + frozen TabPFN 2.5, no per-dataset training. 134 UCR/UEA datasets, 30 resamples; baselines from literature: DrCIF, ROCKET, Arsenal, HC1, HC2 (TSC). No TSFM compared (Mantis only future work).
- TSC mean acc: MultiROCKET-TabPFN 0.8627 vs HC2 0.8674 (69/5/60 wins, p=0.669); vs ROCKET 99/3/32. Runtime vs HC2 2.29x-98.39x faster end to end; energy on CinCECGTorso 102.76 Wh (HC2) vs 1.66 Wh. RTX 3060, CodeCarbon.
- Overlaps: second independent ROCKET+TabPFN pipeline (adds to MASHT/RocketPFN as training-free arm); energy measurement method (CodeCarbon) reusable for Q4.

### B2. TimEE, Kuken, Hoo, Mraz, Hutter, Purucker, arXiv 2607.07500 (Jul 2026) - threat: LOW (but must be a baseline/arm candidate)
- HTML read (first 100k of 136k chars). 4.5M-param in-context learner meta-trained on synthetic VARX tasks (~40 h on one H200). Baselines: Rocket, MiniRocket, Hydra (ridge), DTW-1NN, Catch22, InceptionTime, TS2Vec, and frozen-embedding+RandomForest for NuTime, MOMENT, MantisV2, TiCT, Chronos-2, TiRex. UCR 128 + UEA 24.
- Results: best mean rank on ROC AUC on UCR, third on accuracy; Pareto front of inference speed vs AUC; approaches MiniRocket at ~15-20x faster inference (their claim). Full labeled train set as context, no label budgets; no fine-tuned TSFMs, no scratch control (only multivariate variant updates pooling/ICL/decoder).
- Overlaps: Q4 (cost framing), strong non-ROCKET training-free arm. No fine-tuning comparison.

### B3. TSPFN, Stym-Popper et al., arXiv 2608.31013 (31 Aug 2026) - threat: LOW
- Full text read. PFN adapted to physiological series (7.3M params, ~140k pretraining samples). Baselines at N=100 support samples: XGBoost, TabPFN, TCN, MiniRocket (fine-tuned), LaBraM (EEG FM, fine-tuned). 5 datasets (eICU, ESR, EOS, ECG5000, CPSC).
- Avg AUC: TSPFN 82.5, MiniRocket 78.1, TabPFN 77.5, XGB 77.1, TCN 74.8, LaBraM 64.8. A fine-tuned FM (LaBraM) is far below MiniRocket at 100 labels in this setup. No scratch control, no cost. Text says 5-fold, Fig 3 caption 3-fold.
- Overlaps: Q1 at a single low budget on physiological data; supports "FM fine-tuned below MiniRocket at few labels" for an EEG FM.

### B4. Class-wise MiniRocket, Lo et al., arXiv 2609.07493 (7 Sep 2026) - threat: NONE
- Full text read. MiniRocket variant on 25 UEA datasets; ROCKET-family only; no FM. Baseline accuracies (UEA 25, 30 resamples): HYDRA+MultiRocket 73.35, MultiRocket 73.13, MiniRocket 72.35. Marginal use as a ROCKET-family reference.

## C. Other items checked

- Kim et al., quantum fine-tuning of Chronos-T5 encoder for classification, arXiv 2609.05408 (4 Sep 2026): full text read. Frozen Chronos-T5 base + quantum/MLP head on one dataset (PSML-5, 549 samples). No ROCKET, no scratch control, no label curve, no compute. Threat: NONE.
- Isiacik, Ifrim et al., "An Empirical Evaluation of Foundation Models for Multivariate Time Series Classification", ECML PKDD 2025, DOI 10.1007/978-3-662-72243-5_1 (no arXiv). Only the anthology abstract read (abstract only, unverified): FMs good on simple tasks, weaker as inter-channel dependencies grow; specialised methods win on complex data. Brief item 5.3 said "find another copy": Springer PDF https://link.springer.com/content/pdf/10.1007/978-3-662-72243-5_1.pdf (not fetched). Threat: low (frozen, per abstract).
- Tan Jerome and Simon 2607.04919 (forecasting break-even): already in brief. Search-tool summary mentions ~4 GPU-hours for the whole experiment; I did not verify this in full text.
- 2607.23146 "Foundation Models and Fine-Tuning: Toward a New Generation of Models for Time Series Forecasting": forecasting only (titles/snippets only, abstract only, unverified). Threat: none for classification; possible cost-framing citation.
- TS-LIMITS (NeurIPS 2026 Paris, 13 Dec 2026; https://neurips.cc/virtual/2026/loc/paris/workshop/137623): the listing I saw shows "When to Call a Time-Series Foundation Model: Metric-Dependent Routing and Long-Horizon Limits" (Kadeethum et al.) (title only, abstract not read; forecasting-flavoured). Isiacik 2610.30950 also appears at neurips.cc/virtual/2026/162163. Full accepted list at openreview.net/group?id=NeurIPS.cc/2026/Workshop/TS-LIMITS NOT opened: TODO for Phase 0.
- NeurIPS 2026 tutorial "Time Series Foundation Models in Practice: From Forecasting to Classification and EEG Analysis" listed (title only).
- AIMultiple benchmark: already in brief.

## D. Not found (as of this search)
- Any 2026-06+ paper fine-tuning Chronos/Chronos-Bolt/Chronos-2, TimesFM, Toto for classification against ROCKET-family: none found.
- Any LoRA-on-TSFM classification paper vs ROCKET-family beyond A2: none found.
- Any classification TSFM paper with GPU-hour cost per arm: none found (UniShape's appendix F unread).
- Any second paper with same-architecture from-scratch control across many datasets and label budgets: none found (UniShape ablation: 18 datasets; A2: one dataset).

## Suggested follow-ups
1. Fetch UniShape appendix F (runtime) and its full per-dataset table; check HC2 and whether UCR test-set leakage is controlled (UCR train splits are in pretraining; UCR test not).
2. Open the TS-LIMITS OpenReview group and ACML 2026 FMLTS accepted list.
3. Read the rest of TimEE (last 36k chars) and Springer PDF of Isiacik ECML 2025.
