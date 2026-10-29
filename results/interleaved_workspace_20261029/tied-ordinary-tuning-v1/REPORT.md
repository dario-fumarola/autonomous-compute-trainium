# Tied ordinary independent LR screen

All three1957-update fits, complete cooldowns, mature gradient/state audits and
independent readout checks pass. Each sees128,025,288 authentic targets from
fresh random initialization. Parameters51,179,008; eleven512-wide blocks;
shared32k input/output table. Selection uses the unchanged broad60/20/20 DEV.

| LR | Broad NLL |
|---|---:|
| 0.0003 | 3.608138243 |
| 0.001 | 3.287781024 |
| 0.003 | 3.425804896 |

InteriorLR0.001 is selected. Every fit and failed preparation cost remains
recorded. The shared embedding/output parameter is present once and is exempt
from weight decay; the original independent output table is also exempt. The
historical inventory prose correction changes no optimizer code or result.

This is one-seed reused-DEV tuning, not confirmation. The resulting stronger
control narrows the candidate's equal-data gain to0.025047nats. The candidate
still costs3.782219times its nominal training matrix work.
[Paired result](../tied-comparison-v1/REPORT.md) · [All audits](report.json) ·
[Costs](execution-costs.json) · [Selection](analysis.json).
