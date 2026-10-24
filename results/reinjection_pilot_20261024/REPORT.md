# Completed trained reinjection-only control

| Model | Parameters | Real targets | Independent DEV NLL |
| --- | ---: | ---: | ---: |
| Ordinary | 150,436,608 | 1,004,887,590 | 2.606832753 |
| Trained inexpensive connection | 150,436,608 | 1,004,887,590 | 2.574477293 |

The connection improves prediction by **0.032355460 nats** in seed419, with the
same source order, initial weights, parameters and estimated matmuls. It runs each
block once and adds the prelude state at the core boundary. Extra elementwise
work remains in runtime costs. Both paths use FP32 from random initialization.

Each arm received the same four-LR301M selection budget. Selected peaks were
6e-4 ordinary and1e-3 connection. Both selected TUNE models passed independent
CPU audits; the declared futility gate passed. All six main whole-DEV CPU scores
and serialized full-state copy checks pass. Maximum native/CPU discrepancy is
0.000000716 nats.

This is one development pair, with no seed-significance claim, fresh TEST or
terminal alpha. It gives recurrence a stronger inexpensive baseline. It does
not establish useful additional depth or identify the fraction of the earlier
recurrence gain caused by injection: those fits differ in seeds/precision history.
A common-precision fixed-depth control is still needed for that attribution.

[Exact summary](summary.json) · [Complete LR grid](tuning.json) ·
[Independent audits](independent-cpu-verification.json) · [Costs](execution-costs.json)
· [Frozen protocol](../../docs/REINJECTION_PILOT_PROTOCOL_20261024.md).
