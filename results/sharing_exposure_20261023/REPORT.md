# Sharing and longer-exposure controls

These are single-seed diagnostics following the completed three-seed negative
result. The untied core removes parameter sharing and adds capacity while
preserving the initial two-pass function. Its 251,124,480 unique parameters
compare with 150,436,608 for ordinary/shared paths. It has the same estimated
dense matmul FLOPs as shared recurrence at equal source exposure; optimizer
and communication costs differ. Each architecture has its own equal-budget
three-candidate LR selection. No teacher, synthetic research corpus or RL.

| Arm | Endpoint | Real targets | Parameters | NLL |
| --- | --- | ---: | ---: | ---: |
| ordinary | 1B | 1,004,887,590 | 150,436,608 | 2.518071 |
| shared | 1B | 1,004,887,590 | 150,436,608 | 2.549316 |
| untied | 1B | 1,004,887,590 | 251,124,480 | 2.489247 |

Positive gain below means the candidate has lower NLL than its reference.

| Reference to candidate | Gain | 95% source-group interval, conditional on weights |
| --- | ---: | ---: |
| ordinary_to_untied_1b | 0.028824 | [0.026470, 0.031257] |
| shared_to_untied_1b | 0.060069 | [0.057376, 0.063152] |

All comparisons use identical held-out source targets and the fixed 60/20/20
web/math/code aggregate. The shared/untied 1B contrast tests the sharing plus
capacity constraint; it cannot isolate gradient conflict. The 3B comparison
resumes both untouched stable 900M states, with every Adam moment, RNG, cursor
and past-FLOP counter preserved. Original 64-rank plain-AdamW execution and
selected LRs remain unchanged. The append-only source prefix is validated
byte-for-byte. Additional native qualification updates are discarded before
restarting from the original parents and count as research cost.

Stable checkpoints are descriptive. The 1B and 3B endpoints each complete their
own final decay. These are different horizon branches; a larger exposure does
not establish convergence. A single pair has no estimate of seed variation.
Source-group bootstrap intervals cannot replace independent training seeds.
The original three-seed 1B primary and its negative decision remain unchanged.
All training and readout costs, including tuning and discarded qualification,
are retained separately from reused controls. TEST confirmation is a separate
frozen release after model/analysis identities are fixed.

[Machine-readable readout](summary.json) · [Frozen campaign](../../docs/MECHANISM_DIAGNOSIS_20261023.md).
