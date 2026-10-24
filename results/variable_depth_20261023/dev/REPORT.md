# Variable-depth pretraining: DEV

All natural-source targets and initial weights are paired. All models have
150,436,608 unique parameters. Fixed and mixture training execute exactly the
same estimated neural matmul work; mixture rank depths have probabilities
5/16, 6/16, 5/16 and mean two. Every listed depth is predeclared; no best exit
is selected. Depth4 is outside training support.

| Arm | Seed | Endpoint | Inference depth | Weighted NLL |
| --- | ---: | --- | ---: | ---: |
| ordinary | 103 | 1B | 1 | 2.599135 |
| ordinary | 103 | matched | 1 | 2.495278 |
| fixed | 103 | 1B | 1 | 3.242782 |
| fixed | 103 | 1B | 2 | 2.548269 |
| fixed | 103 | 1B | 3 | 2.611298 |
| fixed | 103 | 1B | 4 | 2.705313 |
| mixture | 103 | 1B | 1 | 2.617777 |
| mixture | 103 | 1B | 2 | 2.587427 |
| mixture | 103 | 1B | 3 | 2.590214 |
| mixture | 103 | 1B | 4 | 2.600341 |
| ordinary | 211 | 1B | 1 | 2.598596 |
| ordinary | 211 | matched | 1 | 2.495164 |
| fixed | 211 | 1B | 1 | 3.134088 |
| fixed | 211 | 1B | 2 | 2.545188 |
| fixed | 211 | 1B | 3 | 2.604903 |
| fixed | 211 | 1B | 4 | 2.695025 |
| mixture | 211 | 1B | 1 | 2.622624 |
| mixture | 211 | 1B | 2 | 2.590665 |
| mixture | 211 | 1B | 3 | 2.593640 |
| mixture | 211 | 1B | 4 | 2.604081 |
| ordinary | 307 | 1B | 1 | 2.601686 |
| ordinary | 307 | matched | 1 | 2.496952 |
| fixed | 307 | 1B | 1 | 3.168541 |
| fixed | 307 | 1B | 2 | 2.546694 |
| fixed | 307 | 1B | 3 | 2.608234 |
| fixed | 307 | 1B | 4 | 2.701771 |
| mixture | 307 | 1B | 1 | 2.618267 |
| mixture | 307 | 1B | 2 | 2.587213 |
| mixture | 307 | 1B | 3 | 2.590489 |
| mixture | 307 | 1B | 4 | 2.601107 |

Positive gain favors the candidate. Primary simultaneous intervals account for three contrasts.

| Primary contrast | Mean gain | 95% seed interval | Family 95% interval | Family decision at 0.01 nats |
| --- | ---: | --- | --- | --- |
| mechanism_mixture_depth1_to_depth3 | 0.028108 | [0.026206, 0.030011] | [0.024727, 0.031490] | USEFUL_GAIN_SUPPORTED |
| architecture_fixed2_to_mixture2 | -0.041718 | [-0.049978, -0.033458] | [-0.056403, -0.027033] | USEFUL_GAIN_RULED_OUT_AT_THIS_EXPOSURE |
| compute_ordinary_matched_to_mixture2 | -0.092637 | [-0.099229, -0.086045] | [-0.104356, -0.080918] | USEFUL_GAIN_RULED_OUT_AT_THIS_EXPOSURE |

The mechanism, architecture and compute claims are separate. Better depth3
than depth1 alone does not establish an efficiency win, learned halting or an
interpretable reasoning algorithm. Student-t intervals have only two degrees
of freedom and assume approximately normal seed differences. This is one
frozen corpus/order. Domain scores, all secondary contrasts, explicit work and
any one-seed3B readouts are in the machine-readable report. Source-group
intervals cannot replace training-seed variation. Estimated training matmuls
include vocabulary projections, but not optimizer/communication/compiler cost.
The old TEST panel is consumed and was not used here. A new intervention after
this study needs another held-out panel.

[Full readout](summary.json) · [Protocol](../../../docs/VARIABLE_DEPTH_PROTOCOL_20261023.md).
