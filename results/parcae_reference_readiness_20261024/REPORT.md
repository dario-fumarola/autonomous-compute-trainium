# CPU reference-readiness checks

Pinned upstream source hashes match the original Git objects. The published
injection operator passes independent value/all-gradient equivalence and finite
differences; the truncated-tail fixture preserves forward values and verifies
the expected input/weight gradients. The full CPU suite passes105 tests.

The code audit identifies reproduction choices around recurrence1 sampling,
dense masked-loop work and tied-output-head accounting. It does not reproduce
the authors' historical training runs or invalidate their reported comparisons.
Only numerical software fixtures were used: no research corpus, held-out scores,
accelerator work or optimizer updates. The audit takes0.732s after imports.
The initial output-directory error and rejected oversized source-transfer call
are preserved in the exact report; their full elapsed time is unavailable.

[Exact report](report.json) ·
[Primary-source interpretation and next decision](../../docs/PARCAE_RINS_REPRODUCTION_TRIAGE_20261024.md).
