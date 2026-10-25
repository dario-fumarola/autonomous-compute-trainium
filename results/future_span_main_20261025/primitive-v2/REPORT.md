# Completed objective pilot and plain-NTP discovery candidate

All eight randomly initialized models complete1,004,887,590 real natural-source
training targets and their own cooldowns. They share initialization,parameters
and source order within each seed. Independent CPU FP32 scoring uses the same
603 consumed-DEV documents,754,750 real targets and fixed60/20/20 weights.
All have166,571,840 training parameters and141,963,840 deployed parameters;
the auxiliary head is removed for scoring. No teacher,generated corpus or RL.

| Condition,declared exit | Seed5231 NLL | Seed7283 NLL | Mean NLL |
| --- | ---: | ---: | ---: |
| One-pass NTP,one | 2.614669 | 2.615013 | 2.614841 |
| One-pass future loss,one | 2.615612 | 2.615944 | 2.615778 |
| Recurrent NTP,two | **2.592239** | **2.589951** | **2.591095** |
| Recurrent future loss,two | 2.590901 | 2.594483 | 2.592692 |

The original future-objective hypothesis fails its declared point gates.
Its recurrence interaction is+0.002281/−0.003602 nats;its improvement over
plain recurrent NTP is+0.001339/−0.004532. Neither seed clears the0.01
interaction threshold. Close this horizon32/stride16/weight0.1 objective
recipe at this budget. Do not replace this primary result with the secondary.
[Complete original analysis and replay manifests](report.json).

The NTP secondary was selected after its full TUNE grid and registered before
MAIN admission. It clears all three development point gates in both seeds:

| Plain-NTP recurrent benefit | Seed5231 | Seed7283 |
| --- | ---: | ---: |
| Over trained one-pass NTP | +0.022430 | +0.025063 |
| Over trained one-pass future loss | +0.023373 | +0.025993 |
| Same trained checkpoint,depth one→two | +0.033283 | +0.031864 |

Its mean equal-data gain is0.023747 nats. Web,math and code gains are positive
in both seeds. A third supported pass adds0.005726/0.005506 nats;the primary
exit remains two. This supports a bounded development mechanism candidate.
Two seeds and conditional source-group intervals do not establish independent
confirmation,training-seed significance or superiority to the strongest
ordinary architecture. [Registered secondary analysis](../../ntp_secondary_discovery_20261025/report.json).

All eight independent full512-window gradient audits pass;the largest overall
body relative-L2 error is0.3181%,below the2% gate. All final model/optimizer
readbacks and32-replica equality checks pass. Selected LRs are.032 for every
cell,at the upper boundary of the shared finite grid;optima remain unbracketed.
The small-core/full-gradient/optimizer/precision recipe differs from earlier
studies,so cross-study changes cannot be attributed to one factor.

Recurrence costs1.232036× nominal training matrix work at equal data. Four
properly cooled one-pass controls at1.238B targets are now admitted. They match
the recurrent budget to0.021527% shortfall,including inherited work. They must
complete before a training-efficiency conclusion. Strong ordinary/cheap
challengers also remain owed. Head-removed native prefill timing is prepared
and queued after these controls;cached generation remains unimplemented.

MAIN alone took14,732.37 seconds,51.700 assigned native chip-hours and65.477
whole-instance opportunity chip-hours. It contains24 native phase closures and
32 CPU audit/scoring attempts summing9,117.08 job-seconds. These parallel CPU
wall durations are neither CPU core-hours nor additional reservation hours.
Qualification,tuning and earlier failures are separate campaign expenditures;
do not present MAIN as the total research cost.
[MAIN costs](execution-costs.json) · [TUNE costs](../../future_span_tune_20261025/primitive-v2/execution-costs.json).

The overall goal remains unachieved. No new TEST or terminal alpha was used.
A terminal claim still requires strong controls,inference costs,new source
groups and at least five fresh paired seeds under a frozen confirmation plan.
