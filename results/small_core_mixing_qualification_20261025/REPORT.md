# Complete cheap-control numerical qualification

Both deterministic-one and sampled mixing paths pass. The transformer core
executes once after the adapter sequence;the inherited `recurrent-ntp` case
key denotes sampled mixing count here. The variant is explicitly registered
as `all_mixes_then_one_core_fullgrad`.

Each isolated32-rank worker executes121 qualification updates,then one
continued update from its complete update120 checkpoint. All replicas retain
identical full model and optimizer state. Every per-rank RNG restores exactly.
Independent pinned-upstream FP32 audits cover all512 original TRAIN windows
at updates0/120,and repeat the resumed point against identical retained gold.
Actual clipping,parameter updates and all optimizer moments pass independent
raw-gradient replay. Maximum body-gradient relative-L2 discrepancy is0.000892.

| Path | 100 warm-step seconds | Real targets/second |
| --- | ---: | ---: |
| One mixing step | 105.944 | 494,011 |
| Sampled mixing count | 108.235 | 483,552 |

These are varying-schedule,actual-size training measurements,including
accumulation and distributed synchronization. Cold setup,captures,recovery,
CPU audits and campaign reservation remain separately charged in
[the receipts](execution-costs.json). Qualification uses immutable qualification-
only random weights and produces no launchable scientific state. There are244
native qualification updates and no scientific updates,held-out score,TEST
access or terminal allocation. [Full numerical report](report.json).
