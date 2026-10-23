# Resource comparison

This records the executed retained training paths, including the ordinary
model's reused pre-annealing history. Estimated matmul matching and measured
execution costs are different quantities.

| Model | Parameters | Real targets | Matmul FLOPs | Training process minutes | Assigned training chip-hours | Process segments |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ordinary-flop-s23 | 150,436,608 | 1,699,926,288 | 1.858147e+18 | 42.40 | 11.307 | 5 |
| untied-1b-s23 | 251,124,480 | 1,004,887,590 | 1.858395e+18 | 39.61 | 10.564 | 2 |
| untied-1b-s47 | 251,124,480 | 1,004,887,590 | 1.858395e+18 | 40.13 | 10.700 | 2 |
| untied-1b-s71 | 251,124,480 | 1,004,887,590 | 1.858395e+18 | 40.29 | 10.745 | 2 |

The ordinary retained prefix ends at its original stable-checkpoint file
timestamp; this reconstructs historical elapsed time and omits that prefix's
terminal teardown. Its five process segments versus two for untied include
different compile/recovery overhead. These sums exclude DEV scoring, CPU audits,
tuning, qualification and idle gaps. The separate campaign execution ledger
retains all of those attempts and reports both accelerator-active time and the
whole observed launch span. Reused controls are not charged a second time to
incremental campaign cost. No claim about total efficiency follows from the
dense-FLOP match alone.

The two-pass path executes about 1.692 times the ordinary forward matmul work
per fixed-context scoring window. These estimates include the vocabulary
projection; they are not autoregressive generation latency measurements.

[Detailed resource ledger](resource-costs.json) · [Learning comparison](REPORT.md).
