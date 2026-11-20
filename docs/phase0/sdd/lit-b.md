# Literature B: three papers from brief 5.3 (read 2026-11-20)

Method note: PDFs were fetched and text-extracted locally with pypdf (paper 1 fully; papers 2 and 3 partly, used to spot-check numbers). Papers 2 and 3 tables were taken from the arXiv HTML renderings via a summarising fetch tool, then spot-checked against the PDF text. Spot-checked and confirmed: paper 2 mean-accuracy block (Fig. 3), 25 datasets, 30 resamples; paper 3 A100/Wilcoxon, Camargo Gyr-Y L3 0.524 -> 0.623 (+9.9), Parkinson rows 0.964/0.974/0.976. Other table numbers below are from the HTML extraction and not individually re-checked. Paper 1's appendix (Tables 7-9, Mantis variants) is NOT in the preprint PDF, so those were not read.

## 1. Isiacik, Le Nguyen, Aderinola, Ifrim (UCD), ECML PKDD 2025
- Citation: P. Sungu Isiacik, T. Le Nguyen, T. Aderinola, G. Ifrim. "An Empirical Evaluation of Foundation Models for Multivariate Time Series Classification." ECML PKDD 2025, pp. 3-21. doi:10.1007/978-3-662-72243-5_1. Code: https://github.com/mlgig/FM4MTSC (cited in text; not opened).
- URL read (full text, 18 pp, main text without appendix): https://ecmlpkdd-storage.s3.eu-central-1.amazonaws.com/preprints/2025/research/preprint_ecml_pkdd_2025_research_1221.pdf . Metadata: https://mlanthology.org/ecmlpkdd/2025/isiacik2025ecmlpkdd-empirical . Springer PDF redirects to login.
- Models: zero-shot (frozen embeddings + RandomForest / LogisticRegression / RidgeClassifier) for Chronos, aLLM4TS, VQShape; fine-tuned: MOMENT, One-Fits-All; Mantis with full fine-tuning (headline row "Mantis Full FT") and, in the appendix (unread), adapters and adapter-head fine-tuning. Text claims adapters cut transform time 33-95%, but reduce accuracy on strongly channel-dependent data (MP50, SYNTH); full FT beats adapter-head FT on those at 15-20% more training time (Sec. 5).
- Baselines: ROCKET, MiniRocket, QUANT, HYDRA, Catch22 (aeon, default params); CNN, TimesNet, InceptionTime, LITEMVTime; ConvTran, TSLANet; RF/GB/KNN/LR/Ridge on flattened series. No HC2, no MultiRocket. No from-scratch / random-init control for any FM.
- Datasets (Sec. 4): only four: CounterMovementJump (CMJ, 3 ch), Military Press MP8 and MP50, SYNTH (8 ch). Splits: not stated beyond using 25% of the training set as validation with early stopping (patience 10, max 100 epochs). Single run per cell, no seeds or CIs reported. Label budget: full training set only.
- Compute (Sec. 5): Apple M1 Pro 16GB; FMs needing fine-tuning on MPS; non-FT methods on CPU. Wall-clock for train/pred reported per cell.
- Key numbers (Table 6 accuracy; CMJ / MP8 / MP50 / SYNTH):
  - ROCKET 0.950 / 0.743 / 0.793 / 0.861; MiniRocket 0.944 / 0.741 / 0.787 / 0.882; QUANT 0.933 / 0.696 / 0.740 / 0.964; HYDRA 0.944 / 0.748 / 0.738 / 0.912.
  - Mantis Full FT 0.955 / 0.697 / 0.773 / 0.929; MOMENT (fine-tuned) 0.855 / 0.489 / 0.662 / 0.768; One-Fits-All 0.866 / 0.644 / 0.266 / 0.672.
  - Best model per dataset: Mantis (CMJ), InceptionTime (MP8, 0.832), ROCKET (MP50), QUANT (SYNTH).
  - Table 2 time (train s / pred s): MiniRocket CMJ 2.26/0.49, MP8 3.58/1.09, MP50 4.86/1.61, SYNTH 119.53/4.88. ROCKET MP50 50.32/20.60.
  - Table 5 fine-tune time (train s / pred s): MOMENT CMJ 81.17/2.17 (50 epochs), MP8 103.52/9.56, MP50 1791.58/119.38, SYNTH 5164.22/61.30. Text: Mantis adapter extraction "seconds", full fine-tuning "hours" (Appendix Table 8 unread, no number).
  - Conclusion quote: MiniRocket "currently still providing more reliable and efficient solutions than foundation models for MTS classification". Text also notes "While foundation models achieve top results, the performance differences are often marginal".
- Limitations: 4 datasets (3 strong channel-dependency), one run, default hyperparameters, no tuning budget parity, FT and zero-shot mixed, no from-scratch control, 'channel dependency' defined by authors' own criterion, FM compute on laptop MPS vs ROCKET on CPU (not comparable hardware).
- Overlap: Q1 (partial, 4 datasets, single seed), Q3 (Mantis full FT vs adapter vs frozen, small), Q4 (runtimes, mixed hardware), Q5 (channel dependency as predictor). None on Q2 or Q6.
- Novelty threat: LOW-MEDIUM. Same direction (Mantis FT vs ROCKET family on MTSC) but tiny, single-seed, no pretraining control, no label budgets. Notable that Mantis FT is the only FM to win (CMJ, SYNTH over ROCKET, MP8 below). Cite as prior art; our design (many datasets, seeds, from-scratch twin, budgets, Pareto cost) is a clear superset.

## 2. Lo, Morvan, Rossi, Morganti, Mercier, arXiv 2609.07493 (v1, 7 Sep 2026)
- Citation: M. M. Lo, G. Morvan, M. Rossi, F. Morganti, D. Mercier. "Improving Multivariate Time Series Classification with Class-Wise Training and Model Aggregation." arXiv:2609.07493v1 [cs.LG], 7 Sep 2026.
- URLs read: https://arxiv.org/abs/2609.07493 , https://arxiv.org/html/2609.07493v1 , https://arxiv.org/pdf/2609.07493 (text-extracted).
- Models: MiniRocket with per-class one-vs-rest classifiers and class-specific channel selection (ECS/ECP elbow selection). No foundation model, no fine-tuning, no pretraining. Variants Base, CW, CW+G, CW+L.
- Baselines: ROCKET, MiniRocket, MultiRocket, HYDRA, HYDRA+MultiRocket only in baseline selection (Sec. 4.2, Fig. 3). No HC2 (mentioned in intro only), no FMs.
- Data: 25 of 30 UEA datasets (equal length, length >= 9), 2-1345 dims, length 30-17,984; each resampled 30 times; scheme and train/test ratio not stated (Sec. 4.1). No label budgets. No compute measurements; only complexity analysis (Sec. 3.4).
- Key numbers: Fig. 3 mean accuracy (%): Hydra+MultiRocket 73.3494, MultiRocket 73.1342, MiniRocket 72.3486, Hydra 71.4718, Rocket 71.3443; W/D/L MiniRocket vs MultiRocket and vs Hydra+MultiRocket both 12/1/12 (Wilcoxon p 0.2940 and 0.6668; which p goes with which pair is ambiguous in the extracted figure). No significant differences among the ROCKET family. Table 1 examples: EthanolConcentration 47.12 -> 49.89; FaceDetection 62.55 -> 65.51. Table 2: UWaveGestureLibrary ECS 70.33 -> 82.94 (CW+L). Gains inconsistent (e.g. ECG down, Table 3).
- Limitations: MiniRocket only; gains mixed; no runtimes.
- Overlap with Q1-Q6: none directly. Useful only as independent evidence that, on 25 UEA datasets (30 resamples), ROCKET-family members are statistically indistinguishable (supports using one strong training-free reference, with MultiRocket/Hydra as sensitivity), and as a possible channel-selection extension of the MiniRocket arm.
- Novelty threat: NONE.

## 3. Coser, Orvieto, Soda, Zollo, arXiv 2608.06122 (v1 6 Aug 2026, v2 7 Aug 2026)
- Citation: O. Coser, A. Orvieto, P. Soda, L. Zollo. "Is Self-Pretraining really useful to improve diagnosis in medical Time Series?" arXiv:2608.06122v2 [cs.LG].
- URLs read: https://arxiv.org/abs/2608.06122 , https://arxiv.org/html/2608.06122v2 , https://arxiv.org/pdf/2608.06122 (text-extracted for spot checks).
- Models: one small Transformer (1/2/3 layers, Sec. III-A). Not a foundation model: self-pretraining (SPT) on the same downstream training data with masked MSE (4 masks: Point, Block, Column, Mixed), then fine-tune (cross-entropy). Control: same architecture from scratch (Xavier init). Mask ratio and block length not reported.
- Baselines: none external; no ROCKET, no HC2, no FM.
- Datasets (Table II): Camargo 2021 (21 subjects, LOSO, 5 classes), Non-EEG Stress (20 subjects, LOSO), Gait Parkinson's (166 subjects, repeated subject-wise splits; counts and ratios unreported). No label-fraction experiments despite "limited data" framing.
- Compute (Sec. IV): one NVIDIA A100 80GB; scratch 10 epochs; SPT 100 epochs lr 0.001; fine-tune 15 epochs lr 0.01. No runtimes. Note unequal training budgets between arms (scratch 10 epochs vs SPT 100 + FT 15), a confound for Q2-style claims.
- Key numbers (accuracy, scratch -> SPT): Table III Camargo Gyr-Y L1 0.532 -> 0.551, L2 0.525 -> 0.578, L3 0.524 -> 0.623 (+9.9pp, p<=0.0001 marker); Parkinson LeftFoot L2 0.745 -> 0.813. Table IV multivariate: Camargo scratch 0.803/0.828/0.806, Column L3 0.869; Parkinson scratch 0.893/0.895/0.918, Block L3 0.976, Column L2 0.974. Table V Mixed: Parkinson L2 0.966 (+7.1). Abstract claims 0-6pp but tables show up to +9.9.
- Limitations: 3 small datasets, unclear Wilcoxon unit, no baselines, unequal budgets, missing hyperparameters, no low-label tests, depth effect not monotone.
- Overlap: Q2 conceptually (pretrained vs same-architecture scratch), but self-pretraining on the target data, not a pretrained FM; no cost, no ROCKET.
- Novelty threat: LOW. Different question (SPT, domain-specific, small transformer).

## Related item found, not in scope, high relevance (abstract only, full text NOT obtained)
- P. Sungu Isiacik, G. Ifrim. "Random Initialization as a Baseline for Time Series Foundation Models." NeurIPS 2026 listing: https://neurips.cc/virtual/2026/162163 ; OpenReview https://openreview.net/forum?id=GQ1J1w8pCJ (blocked by verification page); anonymous code https://anonymous.4open.science/r/tsfm-random-init-baselines-65A7 (not opened).
- Abstract (paraphrased by the fetch tool, so verify wording): 4 published models, 7 instances, random weights vs pretrained, UEA archive, four classifiers on FROZEN representations, 183 paired dataset-level comparisons; mean pretraining gain 2.58pp (range 1.43 to 5.19 across instances); after Holm correction only NuTime significant per instance, pooled analysis significant; ROCKET, Hydra, Quant remain competitive.
- Overlap: Q2 (frozen random-weight twin, which is paper 1's twin in brief). Our Q2 is the FINE-TUNING analogue, so it is not covered, but this is the nearest prior art by the same group. Threat to novelty: MEDIUM (on Q2 framing) until full text is read; recommend reading before Phase 1.

## Summary table
| Paper | Full text | Threat |
|---|---|---|
| Isiacik et al. ECML 2025 | yes (main text, no appendix) | low-medium |
| Lo et al. 2609.07493 | yes | none |
| Coser et al. 2608.06122 | yes | low |
| Isiacik & Ifrim NeurIPS 2026 (extra) | abstract only | medium |
