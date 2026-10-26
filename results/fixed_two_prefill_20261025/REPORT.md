# Completed read-only prefill cost measurement

Both trained primary prediction models pass CPU/native numerical checks with the training-only head removed. Each model is measured on both physical eight-chip halves in two reversed rounds. No weights change,gradients appear or optimizer update runs.

| Round | One median | Fixed2 median | Fixed2 latency increase |
| --- | ---: | ---: | ---: |
| 1 | 22.269ms | 26.190ms | 17.61% |
| 2 | 22.217ms | 25.868ms | 16.43% |

The fixed path uses23.038% more nominal forward matrix work. Maximum observed mean-NLL discrepancy is0.00011330 nats;maximum sampled-logit relative-L2 discrepancy is0.065776%,below the frozen0.003-nat/2% gates.

The stage performs26,112 registered native forwards plus128 independent CPU-reference forwards. CPU preparation takes68.66 seconds. Native assigned cost is0.547304 chip-hours;the separate complete post-MAIN whole-instance opportunity cost is0.981388 chip-hours. Those intervals are not additive. Zero training updates,TEST accesses or terminal alpha.

Scope:preloaded authentic TRAIN windows,1024 positions,full logits,batch1 per core,32 independent cores per eight-chip job,100 warm/100 synchronized timed forwards per rank. This is prefill,not KV-cached token decoding or end-to-end serving. Timing does not establish a favorable deployment frontier or the research goal.

[Exact result](report.json) · [Native costs](native-execution-costs.json) · [CPU references](preparation-report.json) · [Protocol](../../docs/FIXED_TWO_PREFILL_COST_20261025.md).
