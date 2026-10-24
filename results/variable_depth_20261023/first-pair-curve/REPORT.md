# First paired learning curve — exploratory DEV

Seed103 only. Same targets/initialization/parameters; independently annealed endpoints. Group-bootstrap95% intervals condition on these fitted weights and are descriptive, without multiplicity adjustment or a seed-level hypothesis decision. No new fit or TEST scoring; frozen primary analysis is unchanged.

| Comparison | Mixture gain (nats) | Conditional source-group95% interval |
| --- | ---: | --- |
| 100M, equal data | -0.003910 | [-0.006891, -0.000653] |
| 301M, equal data | +0.010161 | [+0.007674, +0.012626] |
| 1.005B, equal data | +0.011708 | [+0.009770, +0.013632] |
| Matched update matmuls | -0.092150 | [-0.095718, -0.089039] |

The final data-matched gain and the compute comparison answer different questions.
The compute control uses1.700B ordinary targets versus1.005B mixture targets.
Earlier points use stable learning rates; endpoints complete their own decay.
No source contents, document/group identifiers or weights are released.

[Figure](first-pair-learning.png) · [Numerical readout](summary.json).
