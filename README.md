# Autonomous computation during pretraining

Can a randomly initialized language model learn to use additional internal
computation to predict unseen natural text better?

Training uses authentic web text, mathematical web text and repository code.
There are no generated research examples, external teachers, reasoning traces
or reinforcement learning. Real-token negative log likelihood (NLL) is the
primary outcome. Equal-data and training-work comparisons answer different
questions and are reported separately.

The latest completed small-core study finds **useful additional computation at
equal data, but no training-efficiency win**. These are two paired development
seeds, with independently scored, cooled endpoints:

| Model | Authentic training targets | Mean DEV NLL ↓ |
| --- | ---: | ---: |
| One pass, next-token prediction | 1.005B | 2.614841 |
| Recurrence, next-token prediction, two-pass exit | 1.005B | **2.591095** |
| One pass, approximately matched training matrix work | 1.238B | **2.581336** |

All have **141.96M deployed parameters**. Recurrence improves equal-data NLL by
0.023747 nats, but loses to longer one-pass training by 0.009759. The work
match is within 0.02153%; it includes the common training-only head and
replicated matrix-optimizer work, and does not imply equal chip-hours.
[Completed comparison](results/future_span_work_controls_20261025/REPORT.md) ·
[Original study](results/future_span_main_20261025/primitive-v2/REPORT.md).

The same recurrent checkpoints improve when evaluated with two supported
passes instead of one. This establishes useful dependence on additional
computation; it does not establish reasoning, learned stopping or superiority
over the strongest ordinary architecture. The proposed future-span auxiliary
objective failed its declared interaction test and is closed.
[Mechanism readout](results/ntp_secondary_discovery_20261025/report.json).

The measured two-pass path takes about **17% longer** than one pass in a
synchronized, preloaded 1,024-position prefill benchmark. Cached decoding and
end-to-end serving were not measured. Even a free, target-informed selector
among these saved whole-document depth paths fails to beat the work-matched
one-pass control.
[Prefill costs](results/future_span_prefill_20261025/REPORT.md) ·
[Oracle bound](results/future_span_depth_oracle_20261025/REPORT.md).

The completed cheap-mixing control repeats the input adapter while executing
the core once. At equal data,it scores **2.613655**, versus recurrence's
**2.591095**, across the same two seeds. It does not recover recurrence's
useful gain. At matched nominal work,cheap mixing scores **2.581267** and
also beats recurrence. Extra core execution contributes useful prediction
work,but still does not earn its training cost under these controls.
[Completed control](results/small_core_mixing_main_20261025/recovery-v2/REPORT.md) ·
[Live status](STATUS.md).

The next investigation audits naturally occurring worked reasoning before
considering a curriculum that hides intermediate steps. This would change
the training data view and add natural procedural supervision. No curriculum
training has launched,and any task-specific result would remain separate from
the broad-text prediction goal.

Earlier studies remain preserved, including the
[original shared/untied TEST comparison](results/frozen_test_20261023/REPORT.md),
[revised fixed/variable-depth TEST comparison](results/variable_depth_20261023/test/REPORT.md),
[longer-exposure compute control](results/mature_fp32_compute_20261024/REPORT.md),
[cheap-connection comparison](results/depth_connection_20261024/REPORT.md),
[small-core pilot](results/parcae_low_depth_pilot_20261024/REPORT.md) and
[fixed-pause negative](results/learning_curve_20261022/REPORT.md).
The previous [1.7B repository](https://github.com/dario-fumarola/useful-thoughts-trainium)
and [archived pause pilot](https://github.com/dario-fumarola/pause-pretraining-trainium)
remain intact. None of the completed recipes has established a competitive
training-compute advantage.

A terminal positive claim requires a frozen candidate, fresh source groups,
at least five fresh paired training seeds, the declared useful-effect threshold
and correction for repeated confirmation attempts. No such confirmation has
been opened for the latest candidate.
[Ongoing charter](docs/ONGOING_RESEARCH_CHARTER.md) ·
[Data rules](docs/DATA.md) · [Trainium notes](docs/OSS_NOTES.md).

Install CPU test and analysis dependencies with
`python -m pip install -e '.[test,analysis]'`, then run
`python -m pytest -q`. Some reference-parity tests additionally require the
pinned upstream checkout specified in their preparation documents. Numerical
software fixtures are not research examples. Source contents, weights, full
optimizer states and native environments stay outside this public repository;
CPU tests do not independently reproduce the scientific training runs.
