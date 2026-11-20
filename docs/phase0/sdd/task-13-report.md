# Task 13 (code + HC2 availability) report
Commit bdb9f08 "Add HC2/MR-Hydra published-results coverage". Files: tor/hc2.py, tests/test_hc2.py, reports/phase0/hc2_availability.md.
- published_results_coverage(meta, classifier, sources=None, fetch=pd.read_csv): reads only names + non-NaN resample counts; exact name matching, no aliasing; first source wins.
- Sources: aeon 1.6.0 get_estimator_results default path (ReferenceResults/classification/accuracy/{HC2,MR-Hydra}_accuracy.csv); HC2 UEA from PublishedResults/HIVE-COTEV2/HC2_TESTFOLDS.csv.
- Coverage: UCR 112/128 for both; UEA HC2 26/30, MR-Hydra 0/30. All 30 resamples. No name mismatches. 16 missing UCR are the variable-length/missing-value sets.
- Tests: tests/test_hc2.py 3 passed (2 mocked, 1 `network` live, which passed when run). TDD: test and module were written together; no separate RED run (deviation).
- Full suite: started but did not finish within time budget (slow, output empty at commit); only focused tests verified.
- Resample 0 = official split: verified only via aeon docstring; bake-off paper text not read (marked unverified in report). tsml-eval GitHub not searched (MR-Hydra UEA file may exist there).

## Fix report
Edited hc2_availability.md: Fungi correction (15 of 16 variable-length/missing; Fungi unexplained, unverified), resample 0 now cited to Bake off redux arXiv 2304.13029 with quotes; UEA HC2 file convention still unverified. No tests affected.
