# Fixed recurrent paths: prospective inference readout

Freeze this diagnostic before any new primary curve scores. It is secondary
and uses curve DEV, not sealed TEST. At each of the three paired 1B ordinary
and recurrent checkpoints, evaluate this matrix without updating any weights:

| Training core passes | Inference core passes | Role |
| --- | --- | --- |
| 1 | 1 | Preserved ordinary endpoint/calibration |
| 1 | 2 | Extra inference recurrence without recurrent training |
| 2 | 1 | Remove the second pass from the recurrent-trained model |
| 2 | 2 | Trained recurrent path/calibration |
| 2 | 4 | Fixed-depth extrapolation diagnostic |

Every condition has exactly the same source tokens, source positions, model
parameters and weights within checkpoint. Only repeated core applications
change. The two calibrated cases must reproduce the original scalar DEV NLL
within 0.0001 nats; every native evaluation first agrees with a CPU forward
and verifies unchanged full state. Shard documents across the 64 qualified
logical cores to use the full instance; each window remains batch 1/context
1024 with the same full-vocabulary objective and real-target mask.

Report all three seeds, domains, source-group intervals conditional on fixed
weights, paired-seed intervals and exact forward matmul estimates. Label
the matrix exploratory with non-simultaneous intervals. A trained two-pass
model losing after removal establishes dependence on its trained path, not
autonomous thought discovery or superiority over the ordinary control. A
fixed two-pass model need not improve when extrapolated to four passes.

If the predeclared 3B seed-23 continuation completes, apply the same matrix
to that one pair, clearly labeled as a single-pair longer-exposure diagnostic.
Run only complete readouts that fit the original campaign deadline; retain
any incomplete cells as missing rather than inventing an aggregate.

Sealed TEST uses a separate release manifest after endpoint weights, protocol
and analysis code are frozen. That manifest lists exact checkpoint hashes,
permitted unroll counts, source hash and evaluator hash. Main confirmation
uses the trained paths only (ordinary 1, recurrent 2); exploratory inference
variants do not acquire TEST-based selection. The evaluator fails closed if
a TEST source or checkpoint/unroll is absent from its release.
