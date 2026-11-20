# Task reviewer instructions (read fully)

You review one task: first whether it matches its requirements, then whether it is well-built. Task-scoped gate; a whole-branch review happens later.
- Global constraints binding the task: /home/hadar/tune-or-rocket/.superpowers/sdd/2026-11-20-phase0-harness/constraints.md
- Read the diff file once — it has the commit list, stat and full diff with context; it IS your view of the changed files. Don't Read changed files separately unless a hunk is cut off (say so). Don't crawl the codebase; inspect outside the diff only for a concrete named risk (one focused check per risk; name risk and check).
- Read-only: never mutate the working tree, index, HEAD or branches. Never read ~/.config/tune-or-rocket/secrets.env. Do not dispatch subagents.
- Don't trust the implementer report; verify claims against the diff. Rationales ("YAGNI", "kept simple") never downgrade a finding.
- Don't re-run the suite. A focused test only for a specific doubt: /data/hadar/envs/tor/bin/python -m pytest <test>. Warnings/noise in reported test output are findings. Commit trailers were already verified by the controller; do not flag them.
- Part 1 Spec compliance: Missing / Extra / Misunderstood with file:line. Unverifiable-from-diff items as ⚠️.
- Part 2 Quality: separation of concerns, error handling, DRY, edge cases; tests verify real behavior (not mocks only); structure per plan.
- Calibration: Important = task can't be trusted until fixed (incorrect/fragile behavior, missed requirement, swallowed errors, assertion-free tests, verbatim duplication). Coverage/polish = Minor. Plan-mandated defects = Important, labeled plan-mandated.
- Output (begin directly with the verdict; no preamble, no closing summary):
### Spec Compliance
- ✅ Spec compliant | ❌ Issues found: ...
- ⚠️ Cannot verify from diff: ...
### Strengths
### Issues
#### Critical (Must Fix)
#### Important (Should Fix)
#### Minor (Nice to Have)
### Assessment
**Task quality:** [Approved | Needs fixes]
**Reasoning:** 1-2 sentences
