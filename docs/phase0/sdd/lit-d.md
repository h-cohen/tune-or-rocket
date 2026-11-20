# lit-d: gap-closing search (2026-11-20)

## Sources accessed / not accessed
- arXiv HTML full text: accessed for every paper marked "full text" below (read through the WebFetch summariser, so numbers are the tool's extraction of the HTML; long pages were truncated at 100k chars where noted).
- arXiv export API (export.arxiv.org/api/query): accessed. This was the main discovery route. Several boolean queries returned 0 hits (over-restrictive), so absence is not proof.
- OpenReview TS-LIMITS group page: NOT accessible (page renders only "Loading", no titles). Substitute: arXiv comment-field search for "TS-LIMITS" (12 entries, 10 are accepted-at-TS-LIMITS). This is NOT the full accepted list (only authors who posted to arXiv).
- NeurIPS virtual page via search: workshop confirmed (Paris, 13 Dec 2026), title "Generalization for Time Series in Tight Settings: Latency, Inference, Memory, Privacy and Sustainability"; no paper titles.
- ACML 2026 FMLTS website: NOT found (two searches, no hit for ACML). Substitute: arXiv comment search "FMLTS" returned exactly one paper, 2611.06988 (Jain et al., already in brief). Full workshop list unknown.
- Semantic Scholar API: NOT usable (HTTP 429 on 3 of 3 attempts, one query "time series foundation model from scratch classification" etc. never returned). Replaced by arXiv API abstract searches; the four requested queries were not run verbatim.
- Isiacik and Ifrim full text (arxiv.org/html/2610.30950): read (full text).

## 1. Isiacik and Ifrim, arXiv 2610.30950 (full text) - threat LOW
- Confirmed: no fine-tuning at all ("The encoder remains frozen in both conditions"; full fine-tuning listed as untested; limitations say fine-tuning may use pretrained information frozen features cannot access). So it does not pre-empt Q1-Q3, and it motivates Q2 (paper-2 from-scratch control after fine-tuning).
- Models: MOMENT S/B/L, Mantis, MantisV2, UniTS, NuTime. Classifiers on frozen features: ridge, logistic regression, RBF-SVM, random forest; hyperparameters by 3-fold stratified CV on train only. 5 random-init seeds {0,7,42,123,2026}. 30 UEA datasets (26-27 paired per model). Wilcoxon + Holm.
- Numbers (pretrained / random / delta pp / p_Holm): MOMENT-S 0.703/0.680/+2.29/0.066; MOMENT-B 0.698/0.684/+1.43/0.233; MOMENT-L 0.702/0.684/+1.76/0.225; Mantis 0.728/0.702/+2.56/0.210; MantisV2 0.721/0.695/+2.58/0.233; UniTS 0.651/0.628/+2.24/0.107; NuTime 0.626/0.574/+5.19/0.002. Pooled +2.58 pp over 183 pairs.
- Cost: no pretraining cost; Appendix E single-thread CPU latency on synthetic input: MOMENT-S 82.48 ms, B 354.91, L 1354.18, Mantis 18.61, MantisV2 15.88, ROCKET 46.83, Hydra 14.26, Quant 28.99 ms. GFLOPs: MOMENT S/B/L 7.4/33.07/119.6 (partial), Mantis 0.874, MantisV2 0.482.
- Stated limitation: comparison with ROCKET/Hydra/Quant is "not architecture-matched".
- Note vs brief: summariser says 26-27 paired datasets per model vs brief's "24 common"; different denominators, not a conflict.

## 2. TS-LIMITS papers found via arXiv comment field (partial list)
Accepted at TS-LIMITS (arXiv comment): 2605.18068 Teger (traffic forecasting), 2610.07430 StaFIR, 2610.35128 STeP-TS (pruning), 2610.30950 Isiacik, 2610.06447 TSFM for predictive control, 2610.08715 prediction-powered inference, 2610.32859 TSFM sustainability, 2610.04547 TMAML, 2610.38021 brain-cancer accelerometry signatures. Under review at it: 2609.08106 (stock prediction). Also 2610.30965 (no workshop in comment).
Only the following touch Q1-Q6:
- 2610.32859 "Sustainability of TSFM Deployment and Pretraining: From Carbon Footprint Analysis to Mitigation via Pruning" (full text, first 100k of 117k chars). Forecasting only (Chronos-2 ~120M, LightGBM, PatchTST); LoRA (2000 steps, lr 1e-4) and full FT (3 epochs) modelled for cost; CodeCarbon + fvcore. No classification, no ROCKET. Useful method template for Q4 (amortised pretraining, per-asset retraining). Threat LOW (Q4 method only).
- 2610.30965 "Energy and Latency First" (full text). Cellular handover binary classification; MiniRocket and catch22 as cheap baselines vs LSTM/Transformer/CNN/LightGBM trained from scratch; no foundation models. LightGBM F1 0.974, CNN-1D 0.914; energy per inference measured on a CPU. Threat LOW (shows cost-first reversal of ranking, supports Q4 framing, not FM-specific).
- Others (Teger, StaFIR, STeP-TS, TMAML, PPI, wearable accelerometry, predictive control): titles/comments only, abstract not read; no sign of classification-FM vs ROCKET comparison. Threat none apparent.

## 3. New / newly detailed papers relevant to Q1-Q6
All full text unless noted.

### 3a. UTICA, arXiv 2603.01348 (2 Mar 2026) - threat LOW-MEDIUM
- Mantis-backbone (6-layer transformer d=256), DINOv2-style self-distillation, pretrained on synthetic data only. Baselines: Mantis 8M, MOMENT 385M, NuTime 2M, GPT4TS. No ROCKET family, no HC2.
- Protocol: linear probe AND full fine-tuning (AdamW, cosine, 100 epochs, lr grid {1e-4,2e-4,1e-3}, 20% train held out as validation, 3 seeds). UCR-128, UEA-21.
- Numbers: UCR linear probe Utica 0.794 vs Mantis 0.792 vs MOMENT 0.779; UCR fine-tune Utica 0.857 vs Mantis 0.850; UEA fine-tune Utica 0.8583 vs Mantis 0.8382; UEA LP 0.7777 vs NuTime 0.7415.
- No label budgets, no from-scratch control, no cost. Overlap: Q3 (FT vs LP gap about +6 pts on UCR: 0.857 vs 0.794). Gives a fine-tuning recipe with validation split and lr grid to copy; does not answer Q1/Q2/Q4. Mantis fine-tuned UCR 0.850 is a datapoint for Q1 against MR-Hydra (UniShape paper: 0.8621).

### 3b. KairosHope, arXiv 2605.18657 - threat LOW (low quality)
- Own model, LP-FT on a filtered UCR subset (fewer than 8 classes, length under 400), mean acc 80.11. No baselines, no label budget, no from-scratch, no cost. Not a threat.

### 3c. TIGER, arXiv 2610.06156 (5 Oct 2026; first 100k of 184k chars) - threat LOW for Q1-Q6, MEDIUM as training-free-arm competitor
- Stacking of QUANT/RDST/WEASEL2/KG-MTP features, 12 base learners, meta-classifier TabICLv2 (frozen) or hard vote when mean samples/class below 60.83. 142 UCR, 30 resamples. Accuracy: TIGER 0.884, KG-MTP 0.876, HC2 0.876, RDST 0.863, QUANT 0.855; +0.0080 over HC2 (p=1.7e-4). Median runtime 145.4 s vs HC2 834.7 s (total 20.65 h vs 149.25 h). Dev subset overlaps evaluation (author's 122-dataset check: 0.889 vs 0.882).
- Implication: the training-free ceiling in the brief (HC2, MR-Hydra, ROCKET+TabPFN) is not the strongest published training-free pipeline; TIGER is a candidate extra arm. No FM fine-tuning, no label budgets.

### 3d. TS2TabPFN, arXiv 2608.04174 (4 Aug 2026) - threat LOW
- catch22 / tsfresh / MultiROCKET features + frozen TabPFN 2.5. 134 UCR/UEA datasets, 30 resamples, baselines from literature (not re-run) except HC2/MultiROCKET-TabPFN runtime. Mean acc 0.8627 vs HC2 0.8674 (p=0.669; W/T/L 69/5/60); vs ROCKET 99/3/32.
- Cost with CodeCarbon (RTX 3060): CinCECGTorso 102.76 Wh HC2 vs 1.66 Wh MultiROCKET-TabPFN; runtime vs HC2 2.29x-98.39x faster end-to-end. Excludes datasets exceeding VRAM; contributions say "189 datasets" vs 134+5 (inconsistent). No FM fine-tuning, no label budget. Relevant to the TabPFN arm's limits and cost reporting (Q4).

### 3e. SwitchPFN, arXiv 2609.29814 - threat LOW
- Frozen TabPFN-v3, 8 UEA datasets, 5 seeds; baselines MiniRocket/MultiRocket/HYDRA+MultiRocket, MultiRocket+TabPFN. SwitchPFN 82.63 +/- 0.45 vs budget-matched MultiRocket+TabPFN 79.09; label efficiency at 25/50/100% of train (68.45/75.40/79.38 validation acc). Small, no FM, no scratch. Shows TabPFN-v3 exists (check against brief's TabPFN version).

### 3f. ChorusTIC, arXiv 2608.24033 (first 100k of 155k chars) - threat LOW
- Training-free in-context classifier pretrained on synthetic episodes. UEA-30 72.27%, UCR-128 81.16%, vs MantisV2+LR 70.50/80.03, UniShape+RF 69.72 (UEA). Label tests: 5/10-shot, 20-60% of train (64.81% at 20% to 69.13% at 60% on UEA). Visible text lists no ROCKET/HC2/fine-tuned FM baselines. Competes as an alternative training-free arm, not on Q1-Q6 directly.

### 3g. TSPFN, arXiv 2608.31013 (MICCAI STACOM workshop) - threat LOW
- TabPFN adapted (7.3M, init from TabPFN weights), further pretrained on 140k physiological series. 5 small datasets, N=100 per task, MiniRocket baseline: avg AUC 82.5 vs MiniRocket 78.1 (but MiniRocket wins ESR 89.8 vs 82.7 and eICU 80.0 vs 79.4). No scratch control, no cost. A "tabular FM adapted, beats MiniRocket at low labels" datapoint; weak protocol.

### 3h. O'Rourke, Trisovic, Bertsimas, arXiv 2606.21776 (causal DAG prior; TabPFN fine-tuning) - threat LOW
- Fine-tunes TabPFN v2.5 on synthetic TSC datasets; 75 UCR/UEA; mean acc .8559 vs .8436 (e=1). No ROCKET baseline, no label budgets, no scratch. Relevant only as evidence that TabPFN itself can be fine-tuned (possible extra arm).

### 3i. Self-supervised PHM, arXiv 2610.35280 - threat LOW
- 0.8M PatchTST, full FT vs linear probe, from-scratch control (single seed, uncalibrated lr, authors call it a lower bound). CWRU macro-F1 scratch 0.350 vs pretrained central-100k 0.812; HSG18 0.569 to 0.951; LP 0.705 / 0.906 on CWRU/HSG18. No ROCKET, no label sweep. 3.58 h on A100 for pretraining. Of note: "backbone lr 1e-4 gives 0.169 and 1e-5 gives 0.829 for the same checkpoint" - supports needing a tuning protocol (Q1/Q2).

### 3j. TS2Vec-FM, arXiv 2610.28602 - threat LOW
- Frozen only; UCR/UEA linear probes (TS2Vec-FM LR 83.32/75.30 vs MantisV2 83.60/75.52). No ROCKET, no fine-tuning.

### 3k. ADAPT, arXiv 2604.08398 - threat LOW
- Many-to-one pretraining, fine-tuned 5 seeds on 4 TF-C benchmarks (Epilepsy, FD-B, Gesture, EMG); baselines are SSL methods, no ROCKET. Not relevant to Q1.

### 3l. OpenWatch, arXiv 2605.04791 (first 100k of 113k chars) - threat LOW
- Fine-tuned NormWear (136M; head-only vs LoRA) loses to a task-specific model: Mix-Token F1 0.903 vs NormWear-LoRA 0.660, NormWear-Base 0.565; Hydra 0.391, InceptionTime 0.624. One dataset, no label budget; note Hydra is worst there (so ROCKET-family is not uniformly strong on wearables). Weak supporting datapoint for Q3/Q5 (LoRA > head-only: 0.660 vs 0.565).

### Seen only as titles/abstracts (abstract only, not read in full)
- 2611.06988 (already in brief). 2607.07500 TimEE (already in lit-a). 2602.00620 TIC-FM, 2511.19694 TiCT, 2510.26777 (in brief).
- 2610.06450 EMG-FM-Bench: nine FMs, 20 EMG datasets, frozen vs fine-tuned and cross-user; abstract only. Possible Q3/Q5 overlap (cross-user shift) but EMG domain only and presumably no ROCKET; worth one full-text read if the wearable domain is included.
- 2606.27672 TSFM embeddings for e-nose (embeddings need fine-tuning; abstract only), 2606.03292 (frozen FM + statistical features, continual), 2609.05408 (quantum head on Chronos), 2411.12824, 2410.03794 FORMED, 2409.12264: abstract only, not relevant to Q1-Q6.
- 2509.18112 "LLMs surpass domain-specific architectures for CTG" - fine-tuned LLMs, "far more compute"; abstract only; not TSC benchmark.

## Overall verdict
- No paper found in TS-LIMITS (partial list), the one FMLTS paper reachable, or the arXiv searches that runs a controlled fine-tuned-TSFM vs ROCKET-family comparison with from-scratch control, label budgets and cost. No HIGH-threat item.
- MEDIUM-ish (unchanged from lit-a): UniShape primary (2601.06429), R2R, Jain et al. UTICA adds a LOW-MEDIUM data point (FT recipe with validation split, FT vs LP gap) but has no ROCKET baseline.
- New implication for the training-free side: TIGER (HC2-beating, 142 UCR) and TS2TabPFN/ChorusTIC/SwitchPFN show the training-free frontier is moving; Phase 0 should decide whether "best training-free pipeline" includes TIGER.
- Gaps remaining: TS-LIMITS and FMLTS official lists unseen; Semantic Scholar unreachable (rate-limited); Nov 2026 papers after 4 Nov only reachable via arXiv API (30-result window, last page returned 2611.06988 and 2611.00670 as newest, so nothing newer matched the queries).
