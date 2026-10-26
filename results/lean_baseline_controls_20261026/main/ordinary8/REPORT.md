# Conventional ordinary8 paired comparison complete

The independently tuned conventional model slightly outperforms fixed recurrence in both reused development seeds at matched nominal training matrix work. This does not establish a useful recurrence advantage over the stronger challenger.

| Seed | Ordinary8 DEV NLL | Fixed-two DEV NLL | Recurrence advantage |
| --- | ---: | ---: | ---: |
| 5231 | 2.482287130 | 2.484647907 | -0.002360777 |
| 17293 | 2.478757173 | 2.483076417 | -0.004319244 |
| Mean | 2.480522151 | 2.483862162 | -0.003340010 |

Weighted losses were independently recomputed from domain loss sums and real-target counts, with exact source-panel and checkpoint identities checked. Each of web, math and code favors ordinary8 in both trained pairs. Conditional source-group intervals for recurrence's advantage are [-0.004785676,0.000091285] and [-0.006561497,-0.002107613]. These describe evaluation-source variation for each fixed trained pair.

The nominal paired-training-seed 95% interval is [-0.015782351,0.009102330]. It spans zero, and two reused development seeds give weak variance evidence on a consumed panel. The 0.01 useful-effect criterion has not been met. This is not a fresh confirmation, a universal recurrence negative or proof of conventional superiority across training seeds. The proposed 4B follow-up has not passed its stronger-control gate.

Both models completed 5,275 updates / 2,760,803,936 authentic targets and their own cooldowns to zero learning rate. Their nominal matrix work is 1,272,538,842,904,800,000 per model, within 0.01187% of the original fixed-two budget. Original candidate training-head work remains charged. This matches nominal operations, not chip-hours.

Both full512-window mature independent gradient audits pass every original 2% family gate. Overall relative discrepancies are 0.370478% and 0.302431%. Final model/state readback, replicas, source counters and exact scoring checkpoint identities pass.

Ordinary8 has eight unique width640 layers, five attention heads, four value-embedding tables and 141,733,120 parameters, below recurrence's 141,963,840 ceiling. It changes width, depth and lookup allocation; it is a practical conventional challenger, not an intervention isolating weight sharing. The independently selected learning rate 0.032 is the top grid edge and remains unbracketed.

Retained scientific training used 11.637053 and 11.642112 assigned chip-hours. Native gradient diagnostics used 0.306820 and 0.305112 separately. Whole-stage reservation opportunity is 25.334432 chip-hours and must not be added again to overlapping assigned time. Compilation, checkpointing and phase overhead remain in their execution receipts. New-challenger inference latency and cached decoding remain unmeasured.

[Independent recomputation](two-seed-analysis.json) · [Full endpoint report](report.json) · [Execution costs](execution-costs.json).
