# Completed mature compute control

| Model | Real targets | Independent DEV NLL |
| --- | ---: | ---: |
| Fixed two passes | 3,014,637,826 | 2.369851103 |
| Ordinary | 5,100,221,387 | 2.342369879 |

Ordinary wins by **0.027481223 nats** in this paired seed103 comparison.
Both completed their own decay, and the ordinary path branches from its untouched
stable2.713B state. Total estimated training matmuls match within0.003051%,
including the vocabulary head. The different BF16/FP32 fractions mean this is
not a matched chip-time comparison. The equal-data recurrence gain remains real;
this longer-exposure efficiency readout remains negative.

Both new whole-DEV CPU FP32 scores and serialized full-state copy audits pass.
This is consumed DEV and one reused seed, with no TEST or terminal claim.

[Exact summary](summary.json) · [Independent audits](independent-cpu-verification.json)
· [Costs](execution-costs.json) · [Protocol](../../docs/MATURE_FP32_COMPUTE_CONTROL_20261024.md).
