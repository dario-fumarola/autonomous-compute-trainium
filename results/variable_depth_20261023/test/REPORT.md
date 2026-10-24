# Variable-depth pretraining: TEST

All natural-source targets and initial weights are paired. All models have
150,436,608 unique parameters. Fixed and mixture training execute exactly the
same estimated neural matmul work; mixture rank depths have probabilities
5/16, 6/16, 5/16 and mean two. Every listed depth is predeclared; no best exit
is selected. Depth4 is outside training support.

| Arm | Seed | Endpoint | Inference depth | Weighted NLL |
| --- | ---: | --- | ---: | ---: |
| ordinary | 103 | 1B | 1 | 2.586054 |
| ordinary | 103 | matched | 1 | 2.482947 |
| fixed | 103 | 1B | 1 | 3.228228 |
| fixed | 103 | 1B | 2 | 2.537610 |
| fixed | 103 | 1B | 3 | 2.600874 |
| fixed | 103 | 1B | 4 | 2.694915 |
| mixture | 103 | 1B | 1 | 2.603944 |
| mixture | 103 | 1B | 2 | 2.573767 |
| mixture | 103 | 1B | 3 | 2.576784 |
| mixture | 103 | 1B | 4 | 2.587041 |
| ordinary | 211 | 1B | 1 | 2.586857 |
| ordinary | 211 | matched | 1 | 2.483433 |
| fixed | 211 | 1B | 1 | 3.120702 |
| fixed | 211 | 1B | 2 | 2.534358 |
| fixed | 211 | 1B | 3 | 2.594686 |
| fixed | 211 | 1B | 4 | 2.684724 |
| mixture | 211 | 1B | 1 | 2.609039 |
| mixture | 211 | 1B | 2 | 2.577666 |
| mixture | 211 | 1B | 3 | 2.581148 |
| mixture | 211 | 1B | 4 | 2.591928 |
| ordinary | 307 | 1B | 1 | 2.590779 |
| ordinary | 307 | matched | 1 | 2.485306 |
| fixed | 307 | 1B | 1 | 3.152199 |
| fixed | 307 | 1B | 2 | 2.533492 |
| fixed | 307 | 1B | 3 | 2.594988 |
| fixed | 307 | 1B | 4 | 2.688016 |
| mixture | 307 | 1B | 1 | 2.605872 |
| mixture | 307 | 1B | 2 | 2.575648 |
| mixture | 307 | 1B | 3 | 2.579283 |
| mixture | 307 | 1B | 4 | 2.590100 |

Positive gain favors the candidate. Primary simultaneous intervals account for three contrasts.

| Primary contrast | Mean gain | 95% seed interval | Family 95% interval | Family decision at 0.01 nats |
| --- | ---: | --- | --- | --- |
| mechanism_mixture_depth1_to_depth3 | 0.027214 | [0.025593, 0.028834] | [0.024333, 0.030094] | USEFUL_GAIN_SUPPORTED |
| architecture_fixed2_to_mixture2 | -0.040540 | [-0.050076, -0.031003] | [-0.057493, -0.023587] | USEFUL_GAIN_RULED_OUT_AT_THIS_EXPOSURE |
| compute_ordinary_matched_to_mixture2 | -0.091798 | [-0.097070, -0.086526] | [-0.101169, -0.082427] | USEFUL_GAIN_RULED_OUT_AT_THIS_EXPOSURE |

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
