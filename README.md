# Autonomous computation during pretraining

Can a randomly initialized language model learn to use additional computation to predict natural text better?

This is a fresh experiment on Trainium2. The first comparison uses ordinary pretraining, fixed pause positions from initialization, and pauses introduced later. Repeated blocks are a subsequent compute control. These mechanisms test extra computation; they do not establish autonomous textual reasoning.

Training and evaluation use natural web text, mathematical web text, and repository code. No generated arithmetic, synthetic rewrites, teacher traces, or RL. Held-out negative log likelihood on real source tokens is the primary metric, reported by domain. Both source-token and FLOP comparisons are required.

The [first qualification report](results/qualification_20261022/REPORT.md) records CPU checks, actual native updates and optimizer recovery, whole-instance throughput, inference timing, and a measured budget. The 150M model runs at about 158k real targets/sec ordinarily and 113k with pauses across all 16 chips. No large scientific run has launched and no pause benefit has been established.

Read the [protocol](docs/PROTOCOL.md), [current status](STATUS.md), [data rules](docs/DATA.md), and [Trainium notes](docs/OSS_NOTES.md). Previous [1.7B evidence](https://github.com/dario-fumarola/useful-thoughts-trainium) and the [closed pause pilot](https://github.com/dario-fumarola/pause-pretraining-trainium) remain preserved.

Run the CPU correctness tests with `python -m pytest -q`. The data preparation script writes source content only outside this repository. Hardware qualification is separate from scientific training.
