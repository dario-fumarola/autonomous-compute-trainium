# Autonomous computation during pretraining

Can a randomly initialized language model learn to use additional computation to predict natural text better?

This is a fresh experiment on Trainium2. The active first comparison uses ordinary pretraining and fixed pause positions from initialization. The late-introduction arm and repeated blocks are follow-ups. These mechanisms test extra computation; they do not establish autonomous textual reasoning.

Training and evaluation use natural web text, mathematical web text, and repository code. No generated arithmetic, synthetic rewrites, teacher traces, or RL. Held-out negative log likelihood on real source tokens is the primary metric, reported by domain. Both source-token and FLOP comparisons are required.

The [learning-curve protocol](docs/LEARNING_CURVE_PROTOCOL.md) is frozen: 150M parameters, ordinary versus early pauses, three paired seeds, and unseen-text NLL at approximately 100M / 300M / 1B real targets. The corpus contains 1.05B natural training tokens, with separate tuning and curve source groups and sealed TEST. Independent LR choices are frozen and the paired main fits are running; see the [live learning-curve report](results/learning_curve_20261022/REPORT.md). No pause benefit has been established yet.

The [packed-trainer report](results/performance_20261022/REPORT.md) records full optimizer recovery, a 64-rank late switch, identical 150M decay-branch replay, fresh DEV evaluation and failure receipts. The latest 100-warm-update measurements across all 16 chips are **855k real targets/sec ordinarily and 696k with causally padded pauses**. Padding improves native kernel compatibility; its full work is counted. The earlier unpadded pause rate was 354k. The [initial qualification](results/qualification_20261022/REPORT.md) remains preserved.

Read the [protocol](docs/PROTOCOL.md), [current status](STATUS.md), [data rules](docs/DATA.md), and [Trainium notes](docs/OSS_NOTES.md). Previous [1.7B evidence](https://github.com/dario-fumarola/useful-thoughts-trainium) and the [closed pause pilot](https://github.com/dario-fumarola/pause-pretraining-trainium) remain preserved.

Run the CPU correctness tests with `python -m pytest -q`. The data preparation script writes source content only outside this repository. Hardware qualification is separate from scientific training.
