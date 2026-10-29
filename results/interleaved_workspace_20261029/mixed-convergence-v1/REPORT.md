# Mixed-round consistency diagnostic — in progress

The first completed endpoint, LR0.0003, passes the unchanged consistency guard
on all six registered TRAIN windows. Its mature K2 and K3 global numerical
audits pass before this readout. No held-out quality score is used here.

| Domain | FixedK2 mean absolute NLL shift to sequential | MixedK2/K3 shift | Mixed forward KL |
|---|---:|---:|---:|
| Web |0.037709|0.005775|0.004939|
| Math |0.073237|0.008060|0.009080|
| Code |0.111294|0.005988|0.006453|

Each mixed-domain shift and KL is below the unchanged0.01 guard. K16/K32 KL is
below1.36e-11. At this same learning rate, fixed-two training failed in every
domain. Architecture, initialization, source exposure, objective and optimizer
are retained; the mixed depth schedule uses more counted training work.

This supports the proposed numerical repair at one setting. It does not show
that quality is preserved or that the ultimately selected learning rate will
pass. The remaining two endpoints and the complete common quality readout are
pending. No scaling or confirmation is admitted from this partial diagnostic.

[First endpoint receipt](mixed-lr0.json) · [Frozen plan](plan.json) ·
[Original comparison](../convergence-v1/REPORT.md).
