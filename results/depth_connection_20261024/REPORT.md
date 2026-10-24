# Common-FP32 depth/connection result

One descriptive seed419 pair on consumed DEV; independent full-panel
FP32 scores and full-state-copy audits pass. No TEST or terminal claim.

| Model | Real targets | DEV NLL | Estimated training matmuls |
| --- | ---: | ---: | ---: |
| ordinary1B | 1,004,887,590 | 2.606832753 | 1.098412e+18 |
| connection1B | 1,004,887,590 | 2.574477293 | 1.098412e+18 |
| fixed1B | 1,004,887,590 | 2.578226554 | 1.858395e+18 |
| ordinaryMatched | 1,699,926,288 | 2.497955532 | 1.858147e+18 |
| connectionMatched | 1,699,926,288 | 2.477058137 | 1.858147e+18 |

Positive differences below favor fixed recurrence:

- Versus ordinary1B: +0.028606199 nats/token.
- Versus connection1B: -0.003749261 nats/token.
- Versus ordinaryMatched: -0.080271022 nats/token.
- Versus connectionMatched: -0.101168417 nats/token.

Each endpoint completes its own decay. Stronger connection/ordinary
controls spend matching estimated matmuls on more natural source text.
This is a development comparison, not evidence across training seeds.
The inexpensive connection is retained; no gain is attributed to another
core execution by subtracting models with different precision histories.

[Exact metrics](summary.json) · [Current execution costs](execution-costs.json) ·
[Reused parent campaign costs](../reinjection_pilot_20261024/execution-costs.json).

Current receipts include discarded LR trials and audits; reused parent work
must not be forgotten. Overlapping intervals are not sequential node time.
Nominal matmul matching is not matched elapsed time or inference latency.
