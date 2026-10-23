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

A separate read-only diagnosis uses twelve previously fixed DEV prefixes, with 512 real targets each. It averages the selected family of corresponding weights between the two trained core copies, without reoptimization. This is a perturbation of one fitted model, not a retrained sharing control or a TEST candidate.

| Projection | Fixed-prefix NLL |
| --- | ---: |
| baseline | 2.335771 |
| project_norm | 2.351442 |
| project_attention | 3.435118 |
| project_mlp | 3.367969 |
| project_all | 3.689310 |

Averaging can change parameter norms and disrupt coadapted features. These effects do not add linearly or establish which constraint caused the training deficit. Separate attention and MLP values matter strongly to this fitted network; this does not rule out learning a better shared solution from scratch. No projected weights are saved or evaluated on TEST.

Exact SVD on every attention/MLP matrix in four preselected, evenly spaced core layers finds rank 16 captures 18.9% of copy-difference energy on average; the median rank for 80% energy is 208. This is weight-space geometry, not a functional compression guarantee or evidence that low-rank adapters cannot learn another solution.
