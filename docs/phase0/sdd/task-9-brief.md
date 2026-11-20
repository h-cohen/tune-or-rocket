### Task 9: FM adapters (one sub-commit per family)

**Files:** Create `tor/models/mantis.py`, `tor/models/moment.py`, `tor/models/chronos.py` (Bolt + Chronos-2), `tor/models/timesfm.py`, `tor/models/toto.py`, `tor/models/reverso.py`, `tor/models/unishape.py`, `tor/models/nutime.py`. Write only the ones Task 8 marked available.

**Interfaces:**
- Consumes: `FMAdapter`, `register`, `Recipe` (Task 7).
- Produces: registered adapters named `mantis8m, mantisv2, moment-small, moment-base, moment-large, chronos-bolt-small, chronos-bolt-base, chronos2, timesfm25, toto, reverso, unishape, nutime`.

For each family, in order Mantis → MOMENT → Chronos → the rest:
- [ ] **Step A: Install** the model's package into the env, then refresh `requirements.lock`.
- [ ] **Step B: Implement the adapter.** `embed` uses the model's own classification or embedding path if it has one (Mantis: the model's output token; MOMENT: the input to its classification head, i.e. mean over patch embeddings). Otherwise it uses the encoder output mean-pooled over tokens (Chronos-Bolt: encoder only, as Jain et al. do). Multichannel data goes per-channel with embeddings concatenated, unless the model is natively multivariate (Chronos-2, Toto). Series longer than `max_length` are resized or truncated the way the model's own code does it; document this in the docstring with a link to the source. `default_recipe()` cites its source. Where none is published, use `Recipe(lr=1e-4, weight_decay=0.01, epochs=100, batch_size=32)` and set `recipe_source="harness default"`.
- [ ] **Step C: Run** `pytest tests/models/test_contract.py -k <name> -m slow -v` on an acquired GPU. Expected: pass for head, lora, full and scratch. A mode the model can't support gets `pytest.skip` with the reason, and that reason goes into the report.
- [ ] **Step D: Commit** `"Add <family> adapter"`.

