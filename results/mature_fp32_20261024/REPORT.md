# Completed paired FP32 learning curve at3B targets

The fixed two-pass model improves prediction at equal natural-source exposure
in this one continued pair. Both endpoints completed their declared decay.

| Model | Parameters | Real targets | Independent DEV NLL |
| --- | ---: | ---: | ---: |
| Ordinary | 150,436,608 | 3,014,637,826 | 2.405907043 |
| Fixed two passes | 150,436,608 | 3,014,637,826 | 2.369851103 |

The descriptive gain is **0.036055940 nats per real token**. This is one reused
paired seed103 on consumed DEV, not a seed-robust significance claim, fresh
confirmation, pure timing study or completed compute-efficiency comparison.
Both histories are BF16 below2.010B, then the same gradient-qualified FP32
execution policy. No architecture, peak LR, loss, source order or endpoint changed outside the
registered schedule.

| Domain | Ordinary NLL | Fixed NLL | Ordinary minus fixed |
| --- | ---: | ---: | ---: |
| Web | 2.879228 | 2.838200 | +0.041028 |
| Math | 2.091221 | 2.056372 | +0.034849 |
| Code | 1.300630 | 1.278284 | +0.022345 |

All six new whole-panel independent CPU FP32 scores pass the0.0002-nat
agreement gate; maximum native/CPU difference is0.000000808.
All six serialized full optimizer/model copy audits pass. Every training phase
has identical final replica states. The37 frozen source
hashes remain unchanged. Scoring covers603 documents and754,750 original targets,
with fixed60/20/20 web/math/code weights.

The stable2.211B gain is0.028051 and the stable2.713B gain0.023699. The3.015B
completed-decay gain is0.036056. Stable and annealed points are distinguished;
these observations do not establish a monotonic or asymptotic crossover.

Fixed uses **1.691892×** estimated training
matmuls at equal data, including the vocabulary head. The separately registered
ordinary5.100B completed-decay control is now training from its stable2.713B
checkpoint. Its total estimated matmuls differ from fixed3B by0.003051%.
That result is pending. No compute-efficiency win is inferred here.

The new paired execution took2.072 hours including the CPU tail.
Its12 closed native training/evaluation attempts assigned
30.175 chip-hours. These are this execution's
incremental costs, excluding the parent2B histories and prior qualification/data
campaigns, which have separate preserved ledgers. Concurrent campaigns overlap;
wall intervals and per-job opportunity estimates must not be summed as sequential
reservation time. Recovery copies are on this instance, not off-instance backups.

[Independent audits](independent-cpu-verification.json) · [Exact summary](summary.json)
· [Execution costs](execution-costs.json) · [Frozen protocol](../../docs/MATURE_FP32_PROTOCOL_20261024.md)
· [Standalone curve](figures/paired-learning-curve.png)
([SVG](figures/paired-learning-curve.svg), [PDF](figures/paired-learning-curve.pdf)).
