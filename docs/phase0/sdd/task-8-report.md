# Task 8 report: model availability survey
Implemented tor/availability.py (Candidate with weights_url + notes, 15 CANDIDATES, check(), survey()), tests/test_availability.py (8 tests, HF/HTTP mocked), reports/phase0/models_availability.csv (15 rows). Full suite: 65 passed. Real run `python -m tor.availability`: all 15 rows status ok; none gated, none not_run.
TDD: tests and implementation were written together (RED step skipped); one test-side failure (httpx vs httpx2 in constructing GatedRepoError) fixed, then 8 passed.
Findings: UniShape weights are in GitHub (pretrained_model_ckpt/*.pth), no LICENSE (licence None). NuTime ckpt in GitHub, MIT. Reverso 2.6M = shinfxh/reverso-base (inferred from config 8 layers d=128; README stale); reverso-small = shinfxh/reverso-small. HF cards for both lack licence; GitHub is MIT. Toto = Toto-Open-Base-1.0 (Toto-2.0 family exists, not surveyed).
Not committed: runs/runlog.jsonl (15 availability records written by the real run).
