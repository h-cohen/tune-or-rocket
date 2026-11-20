# lit-c: R2R paper details, UniShape, NuTime, Reverso (checked 2026-11-20)
Caveat: WebFetch returns model-summarised text, not raw. R2R appendix A.3 was read in full via fetch offset 100000; quoted numbers should be re-checked against the HTML before citing.

## 1. R2R (arXiv 2610.26114)
### (a) Main table (Sec 5.1, 128 UCR, fully supervised, official UCR train/test splits). Avg rank / Wins
Rocket 7.62/18; MiniRocket 7.75/13; RDST 7.38/20; MR-H 6.27/22; HC-2 5.16/31;
InceptionTime 9.41/13; TS2Vec 12.65/8; PatchTST 18.19/6; TimesNet 17.11/5; SoftShape 9.20/14; MambaSL 13.13/14;
GPT4TS 17.06/4; MOMENT 17.58/2; UniTS 16.74/2; NuTime 10.26/15; TS-RAG 9.31/11; Mantis 8.07/21; UniShape 4.47/31;
R2R(NuTime) 7.90/23; R2R(Mantis) 6.18/29; R2R(UniShape) 3.01/48.
21 methods = 18 baselines + 3 R2R variants. Wilcoxon p-values vs R2R(UniShape) also given (e.g. UniShape 1.68E-04, HC-2 3.62E-07, MR-H 2.04E-10).
### (b) Ablation table (Sec 5.3, "Table 2"; same 128 UCR datasets per caption; App D Table 10 per-dataset)
R2R(UniShape) 1.48 (86 wins); w/o feature transfer 2.45 (48); w/o param transfer+retrieval, MiniRocket 4.14 (13); w/o retrieval MiniRocket 5.09 (11); w/o ReMGP 5.59 (8); w/o retrieval Rocket 5.91 (8); w/o retrieval MultiRocket 4.06 (18); w/o retrieval UniShape(frozen backbone) 4.52 (13).
### Why they differ
- Text does NOT state how ranks are computed. Inference (labelled as such): the ablation ranks only its 7 rows against each other (same 128 datasets), whereas the main table ranks 21 methods. Different method sets, not different datasets. Ranks are per-dataset test accuracy ranks averaged over datasets (main text: avg rank + Win count on test accuracy).
- Corroboration: "w/o feature transfer" has p=1.68E-04, identical to fine-tuned UniShape's p in the main table, so it is likely the same fine-tuned-UniShape run; 1.48 vs 3.01 for R2R(UniShape) then reflects field size. Unverified.
- Wins: R2R(UniShape) 48 in main table vs 86 in ablation; the 86 is consistent with fewer competitors.
- Inconsistencies in paper: text sometimes calls ablation "Table 3"; CD-diagram caption says "158 datasets" vs 128 in text. Ablation Table 11 (KB size) uses a subset of 18 UCR datasets; not the rank table.
### Mean accuracies
Not reported for any method. Appendix B Table 8 gives per-dataset accuracies only; its summary rows are Avg. Rank and Win (no mean-accuracy row). Per-dataset values for UniShape, Mantis, NuTime, MR-H, HC2 are therefore available only by transcribing Table 8 (not done; paper has no machine-readable table seen).
### Fine-tuning recipe (App A.3)
- Epochs: max 500 for R2R+NuTime/Mantis; UniShape up to 300.
- Optimizer, lr, batch size: NOT stated. Claim: baselines and R2R use each method's officially released hyperparameters, same hardware budget, no extra tuning.
- Model selection: minimum training loss (no default validation set). Gradient-free backbone chosen per dataset via validation split of train set.
- Other: downsample {1,2,4}, window 11, K=20 retrieved neighbours; five independently trained models aggregated (ensemble); reported = average test accuracy over 5 runs; seed values not given.
- Hardware: 2x NVIDIA RTX A6000, Ubuntu 20.04.3, Python 3.9, PyTorch 1.12.
- Splits: official UCR train/test. Knowledge base from pretraining data of the three TSFMs (1.89M), class-balanced 100K subset.
- Implication: baseline numbers appear to come from each method's own ensemble/recipe (5-run ensembles), so not a controlled single-run comparison. No GPU-hours.

## 2. UniShape
- Paper: "A Unified Shape-Aware Foundation Model for Time Series Classification", arXiv 2601.06429 (10 Jan 2026), AAAI 2026 (vol 40(28) pp 23972-23980). Zhen Liu et al., Qianli Ma lab.
- Code: https://github.com/qianlima-lab/UniShape (arXiv page itself lists no code link). Scripts: unishape_pretrain.py, unishape_finetune.py, unishape_zeroshot.py, models/, utils/, datapre/, pretrained_model_ckpt/ (contents unverified), supplementary PDF. Fine-tune cmd: `python unishape_finetune.py --dataset CBF ...`. So a classification fine-tuning script EXISTS.
- Weights: repo has a `pretrained_model_ckpt/` folder; README gives no download link for weights (only dataset links on Google Drive). Whether the folder contains a .pth: UNVERIFIED, check by listing the folder (Phase 0 task). Paper does not mention weights.
- Licence: none visible in README/sidebar (UNVERIFIED; likely no licence file -> all rights reserved, flag).
- Params: 3.1M (Table 1).
- Input length: all series resized to fixed 512 via PyTorch interpolation; multivariate split into univariate channels; shape-aware adapter slices into windows at 5 scales (64,32,16,8,4; window=stride).
- Pretraining corpus: 1.89M univariate samples = training sets only of UCR (128), UEA (30, split univariate), plus 8 TFC datasets. YES includes UCR/UEA TRAIN splits (test excluded); the 30 extra eval datasets excluded. Overlap with downstream UCR train not discussed. Same corpus as NuTime's (1.89M).
- Recipe: pretrain <=30 ep, batch 2048, 10% labels; fine-tune up to 300 ep, aux loss weight 0.01, CE, lowest-train-loss checkpoint, 5 seeds. Optimizer/lr/fine-tune batch not in paper main text (in Appendix A / code, unchecked).

## 3. NuTime
- Paper: arXiv 2310.07402, TMLR 2024. Code: https://github.com/chenguolin/NuTime (README also mentions a microsoft/NuTime repo; canonical one unconfirmed).
- Licence: MIT. Weights: ckpt/checkpoint_bias9.pth in repo (BYOL self-supervised, 9 multi-scaled embeddings).
- Params: not stated. Backbone: 6-layer, 8-head Transformer, d=128, MLP 512 (compute from this if needed).
- Pretraining: same 1.89M univariate corpus; UCR/UEA TRAIN splits included, test excluded.
- Fine-tuning: README only shows Epilepsy demo `python3 src/pipeline.py --config_file configs/demo_ft_epilepsy.json`; configs for UCR not listed (only Epilepsy dataset shipped, preprocess.py handles UCR/UEA). Paper: full FT default 100 ep, lr 2e-4; UCR vs supervised baselines = ensemble of 5 runs x 500 epochs. Optimizer/batch for FT not stated (pretrain: AdamW, bs 2048).

## 4. Reverso
- Paper: arXiv 2602.17634 "Reverso: Efficient Time Series Foundation Models for Zero-shot Forecasting" (Fu, Li, Papaioannou, Kim; ICML 2026). Code https://github.com/shinfxh/reverso, MIT.
- Weights: Hugging Face shinfxh/reverso, but README says only Reverso-Small (550K) is available at checkpoints/reverso_small/; the 2.6M Reverso and 200K Nano are "Coming soon" (as of fetch). IMPORTANT for the brief's "Reverso 2.6M" row: 2.6M weights possibly not public; re-check HF.
- Params: 2.6M (8 layers, d=128) / 550K / 200K. Long conv + DeltaNet hybrid.
- Pretraining: GiftEval pretrain data (capped ~100k samples per dataset/epoch) + 1M synthetic series (GP kernels, spikes, TSI). UCR/UEA not in pretraining per paper (paper used them for classification evaluation; only first 100k chars read).
- Fine-tuning script: none in repo (only forecast_demo.py, eval_gift.py). Classification FT path must be built by us.
