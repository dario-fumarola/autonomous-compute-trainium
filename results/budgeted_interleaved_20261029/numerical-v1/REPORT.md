# Explicit thought-budget hardware qualification passes

All five registered graphs pass cold independent CPU gradient checks on the
actual51.747M shape and16-chip/64-rank topology. Full-state restart reproduces
the largest graph's update104 exactly; every replica agrees. The complete
128-window mature audit has maximum family relativeL2 error1.219925e-6 and
loss discrepancy1.780689e-6, within unchanged numerical tolerances.

| Graph ID | Latent tokens | Numerical rounds | Repeated update mean, seconds |
|---|---:|---:|---:|
| 0 | 0 | 0 | 0.153467 |
| 12 | 1 | 2 | 0.246898 |
| 13 | 1 | 3 | 0.279812 |
| 24 | 2 | 4 | 0.461214 |
| 25 | 2 | 5 | 0.526315 |

Every graph meets the0.8-second registered repeated-update ceiling. First use
of each graph and all cold compilations remain in total costs. The two Native
attempts consume3.496071assigned chip-hours; the independent mature CPU audit
takes130.163seconds. All states are disposable and excluded from science.

This is numerical feasibility, not a trained multi-budget model or a quality
result. No DEV or fresh reserve was read. The fixed-one replication gate remains
failed, and the prepared ephemeral scientific comparison remains unlaunched.

[Full numerical receipt](report.json) · [Execution costs](execution-costs.json).
