# Autonomous computation during pretraining

Can a randomly initialized language model learn to use additional computation to predict natural text better?

The active Trainium2 study compares ordinary pretraining with a fixed, differentiable recurrent core from initialization. Its single-pass path is exactly the qualified ordinary architecture. These mechanisms test extra computation; they do not establish autonomous textual reasoning.

Training and evaluation use natural web text, mathematical web text, and repository code. No generated arithmetic, synthetic rewrites, teacher traces, or RL. Held-out negative log likelihood on real source tokens is the primary metric, reported by domain. Both source-token and FLOP comparisons are required.

The [recurrent protocol](docs/PRELN_RECURRENT_PROTOCOL.md) freezes 150M parameters, three paired seeds, a 1B real-target primary endpoint and a separately annealed equal-FLOP comparison. The completed [learning curve](results/recurrent_learning_20261023/REPORT.md) favors ordinary: **2.5199 NLL** versus **2.5504** with recurrence, gain **-0.03054 [-0.03220, -0.02888]**. Recurrence also loses at matched estimated FLOPs. A separate FP32 CPU implementation verifies all six terminal scores. The longer continuation and TEST confirmation did not run; TEST remains sealed. [Current status](STATUS.md).

The completed [same-weight diagnostic](results/recurrent_mechanism_20261023/REPORT.md) shows that recurrent-trained checkpoints depend on their second pass: NLL is **3.0503 / 2.5504 / 2.6933** at one/two/four passes. Applying two passes to ordinary checkpoints worsens NLL. A new [bounded campaign](docs/MECHANISM_DIAGNOSIS_20261023.md) qualifies the untied sharing control and a single-pair 3B continuation, using a persistent stage queue.

The previous [fixed-pause study](results/learning_curve_20261022/REPORT.md) is complete: at 1B targets ordinary scores **2.520 NLL** and pauses **2.574**, gain **-0.0539 [-0.0603, -0.0475]**. The [position diagnostic](results/pause_locality_20261023/REPORT.md) finds losses in every bin, largest immediately after a pause. Its original protocol, data, models and negative decision are preserved. The campaign used 3.69 node-hours; it is not charged again as new training.

The [existing 1.7B checkpoint readout](results/transfer_probe_20261023/REPORT.md) finds initial trace-versus-raw exam NLL gains, which disappear after common linear-readout adaptation. This is a frozen-feature probe, with all models and references retained; full backbone fine-tuning remains untested.

The [packed-trainer report](results/performance_20261022/REPORT.md) records full optimizer recovery, a 64-rank late switch, identical 150M decay-branch replay, fresh DEV evaluation and failure receipts. The latest 100-warm-update measurements across all 16 chips are **855k real targets/sec ordinarily and 696k with causally padded pauses**. Padding improves native kernel compatibility; its full work is counted. The earlier unpadded pause rate was 354k. The [initial qualification](results/qualification_20261022/REPORT.md) remains preserved.

Read the [reproduction guide](docs/REPRODUCE_CURVE.md), [protocol](docs/PROTOCOL.md), [current status](STATUS.md), [data rules](docs/DATA.md), and [Trainium notes](docs/OSS_NOTES.md). Previous [1.7B evidence](https://github.com/dario-fumarola/useful-thoughts-trainium) and the [closed pause pilot](https://github.com/dario-fumarola/pause-pretraining-trainium) remain preserved.

Run the CPU correctness tests with `python -m pytest -q`. The data preparation script writes source content only outside this repository. Hardware qualification is separate from scientific training.
