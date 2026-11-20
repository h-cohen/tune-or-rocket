# Lessons from reviews of finished adapters (all REQUIRED)
Best references: tor/models/timesfm.py, tor/models/chronos.py (+ tests/models/test_timesfm.py, test_chronos.py).
1. Verify pretrained weights really load: test compares a state_dict tensor with the checkpoint file; load with strict=True where possible.
2. Report total params AND params used by the classifier: remove unused decoders/forecast heads/embeddings from classifiers; expose `unused_param_count()`; test full == total − unused + head. If the model name/card states a size, explain any gap by measured subtraction in the docstring (no unexplained gaps, no unverified claims).
3. Reuse the package's own input pipeline (scaling, patching, masks, RevIN) and add an exact-equality test (torch.equal or atol 0) between the adapter's hidden states and the package's own encoder/prefill output.
4. Pool over real patch tokens only (exclude special/summary/REG/future tokens and fully padded patches); mean pooling, documented as a harness choice unless a cited protocol exists.
5. Never state an unverified claim as fact: mark "inferred"/"not verified"/"not documented".
6. preprocess raises ValueError on NaN/inf input.
7. Tests: scratch differs from pretrained, same seed reproduces, different seeds differ; head mode has 0 trainable backbone params; lora trainable params are only lora_ on the attention targets; short input and constant input give finite embeddings.
8. Install with --no-deps if the package would downgrade torch/transformers/peft; verify the adapter's import paths need only installed deps.
9. Leave no background processes running when you report.
