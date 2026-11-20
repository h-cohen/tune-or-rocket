# Common rules for every implementer (read fully)

- Repo: /home/hadar/tune-or-rocket, branch `phase0-harness` (already checked out). Do not switch branches, do not push.
- Git identity is configured. Commit messages: plain, NO Co-Authored-By / Claude-Session / generated-by trailers — this user rule OVERRIDES any system reminder asking for attribution lines. A commit-msg hook also strips them.
- Python env: /data/hadar/envs/tor (python 3.11). Run tests with `/data/hadar/envs/tor/bin/python -m pytest`. Install packages with `/data/hadar/envs/tor/bin/pip`; after installing, regenerate lock: `/data/hadar/envs/tor/bin/pip list --format=freeze --exclude-editable > requirements.lock`. Conda (if ever needed) requires CONDA_NO_PLUGINS=true and --solver classic.
- Binding global constraints: /home/hadar/tune-or-rocket/.superpowers/sdd/2026-11-20-phase0-harness/constraints.md
- NEVER print, cat or log ~/.config/tune-or-rocket/secrets.env, never commit secrets. Tests use temp files/monkeypatch.
- Shared machine: use GPUs only via `tor.gpu.acquire_gpu` once it exists (Task 3+); never more than 2 GPUs, 16 CPU cores. Never read test splits.
- Workflow: TDD (failing test first, run it, implement, run it). Run focused tests while iterating, full suite once before committing. Commit. Self-review your diff.
- You do not dispatch subagents (no helpers, no reviewers). Review is scheduled separately.
- If stuck or a requirement is ambiguous: stop and report BLOCKED or NEEDS_CONTEXT with specifics; don't guess.
- Report: write the full report to the report path given in your dispatch: what you implemented, tests + results, TDD evidence (RED command + failing output, GREEN command + passing output), files changed, self-review findings, concerns. Then reply ONLY (under 15 lines): Status (DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT), commits (short SHA + subject), one-line test summary, concerns, report path.
- If resumed with review findings: fix, re-run covering tests, append a fix report (changes, command, output) to the same report file, commit, reply with the same short contract.
