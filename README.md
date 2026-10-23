# Autonomous computation during pretraining

Can a randomly initialized language model learn to use additional computation
to predict unseen natural text better?

The [ongoing research charter](docs/ONGOING_RESEARCH_CHARTER.md) records the
owner's multi-day authorization, independent-confirmation standard and repeated-
testing rules. Individual experiments remain bounded and falsifiable.

A separately [predeclared variable-depth follow-up](docs/VARIABLE_DEPTH_PROTOCOL_20261023.md)
has passed qualification and completed its300M-target LR screens. Three-seed
main fits now use all16 chips, with fresh held-out groups and an explicit common
optimizer recipe. No primary endpoint result is established yet; the
completed results below remain unchanged. [Live status](STATUS.md).

The completed Trainium2 study compares ordinary pretraining with a fixed,
differentiable recurrent core available from initialization. Training and
evaluation use natural web text, mathematical web text and repository code.
There are no generated research examples, teacher traces or RL. Held-out
negative log likelihood on real source targets is primary; equal-data and
estimated training-compute comparisons are reported separately.

The [sealed twelve-model TEST confirmation](results/frozen_test_20261023/REPORT.md)
is complete. At 1.005B targets, three paired seeds give:

| Model | Unique parameters | Mean TEST NLL, lower is better |
| --- | ---: | ---: |
| Ordinary | 150.4M | 2.578753 |
| Shared two-pass | 150.4M | 2.610307 |
| Untied two-pass | 251.1M | 2.549910 |

Shared recurrence loses by **0.031554 nats/token**, with training-seed 95%
interval **[0.024888, 0.038220]** for its deficit. Untying improves every
observed equal-data run, but adds 67% more parameters. Its mean gain is
**0.028842 [0.002752, 0.054932]**, inconclusive at the predeclared 0.01-nat
useful-gain threshold. This combined interval includes the discovery seed;
the separate two-fresh-seed interval is much wider.

At nearly matched estimated dense training matmul work, a separately annealed
ordinary seed23 endpoint at 1.700B targets beats untied at 1.005B by
**0.071734 [0.065560, 0.077985]** on TEST. This is a single fitted pair and
does not match wall time exactly. A separate unchanged 3.015B ordinary/shared
continuation is nearly tied: **2.391002 / 2.393539 TEST NLL**, also one pair.
**This recipe has no demonstrated compute-efficiency advantage.**
[Resource accounting](results/confirmation_followup_20261023/RESOURCES.md) ·
[Longer-exposure curve](results/sharing_exposure_20261023/learning-curve-seed23.png).

The [original three-seed DEV study](results/recurrent_learning_20261023/REPORT.md),
[same-weight mechanism diagnostics](results/recurrent_mechanism_20261023/REPORT.md),
[untied replications](results/confirmation_followup_20261023/REPORT.md) and
[one-seed longer continuation](results/sharing_exposure_20261023/REPORT.md)
remain preserved. The shared model depends on its trained second pass;
further repeated passes hurt. The untied control starts with the same two-pass
function, then learns separate core weights. It tests sharing plus capacity,
not autonomous reasoning or learned allocation of computation. Independent
full-panel FP32 audits, all-replica equality and full optimizer recovery pass.
Checkpoint identities and evaluation sources were frozen before TEST access.

After scientific scoring closed, [software qualification](results/fast_path_20261023/REPORT.md)
validated unscaled fused AdamW and two concurrent eight-chip jobs, each with
100 warm updates: **874k ordinary / 576k shared real targets/sec**. The actual
paired completion rate is **1.152M targets/sec**; the individual rates must not
be summed. These are warm qualification rates without periodic scientific
checkpoint overhead. A separate [operator-contract reproducer](docs/OSS_FUSED_ADAMW_METADATA_20261023.md)
documents ignored optional AMP metadata; neither the completed science nor
the qualified unscaled candidate uses it. No upstream submission was made.

The earlier [fixed-pause study](results/learning_curve_20261022/REPORT.md)
and [position diagnostic](results/pause_locality_20261023/REPORT.md) are
preserved negative results. The [existing 1.7B checkpoint readout](results/transfer_probe_20261023/REPORT.md)
finds initial trace-versus-raw exam gains that disappear after common
linear-readout adaptation; full backbone fine-tuning remains untested.
Previous [1.7B evidence](https://github.com/dario-fumarola/useful-thoughts-trainium)
and the [closed pause pilot](https://github.com/dario-fumarola/pause-pretraining-trainium)
remain intact.

Read [current status](STATUS.md), the [recurrent protocol](docs/PRELN_RECURRENT_PROTOCOL.md),
[follow-up protocol](docs/CONFIRMATION_FOLLOWUP_20261023.md),
[reproduction guide](docs/REPRODUCE_CURVE.md), [data rules](docs/DATA.md),
[execution ledger](results/sharing_exposure_20261023/execution-costs.json) and
[Trainium notes](docs/OSS_NOTES.md). The fixed-depth campaign is complete. The variable-depth follow-up uses its
own fresh panel and bounded protocol. Further variants require another held-out panel.

Run CPU correctness tests with `python -m pytest -q`. Source contents, weights
and full optimizer states stay outside this public repository.
