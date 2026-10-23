# Fixed recurrence: completed three-seed natural-text study

At **1,004,887,590 real training targets per model**, the ordinary model wins
in all three paired initializations and all three domain means. Ordinary NLL
is **2.519907**, versus **2.550448**
with the two-pass shared recurrent core. The gain, defined as ordinary minus
recurrent NLL, is **-0.03054** with paired-seed 95% interval
**[-0.03220, -0.02888]**.
The predeclared 0.01-nat useful gain is ruled out for this recipe and exposure.

| Real targets/run | Ordinary mean NLL | Recurrent mean NLL | Ordinary minus recurrent, 95% seed interval |
| --- | ---: | ---: | ---: |
| 100,488,104 | 3.75159 | 3.99059 | -0.23900 [-0.30916, -0.16884] |
| 301,465,908 | 2.97414 | 2.99425 | -0.02011 [-0.03381, -0.00642] |
| 1,004,887,590 | 2.51991 | 2.55045 | -0.03054 [-0.03220, -0.02888] |

![Learning curve](learning-curve.png)

The 100M and 301M points are descriptive stable checkpoints. The primary 1B
endpoints both complete their cosine decay. The deficit shrinks sharply early,
but the observed points do not justify a prediction of a later crossover.

Both paths have **150,436,608 unique parameters**, identical paired initial
weights, natural source targets/order, tokenizer, global batch and AdamW recipe.
Ordinary controls are reused explicitly from the preserved completed study.
Actual CPU outputs and full gradients prove the single-pass recurrent wrapper
is equivalent to the healthy ordinary architecture. Two shared core passes
produce 36 block applications versus 20 ordinary blocks. There is no external
teacher, generated research corpus, RL, textual thought or learned halting.

Independent, equal-budget 100M-target LR selection chose 3e-4 for recurrence
from 3e-4/1e-3/3e-3 (tuning NLL 4.07541/4.23321/4.85068). The ordinary reference
selected 1e-3 with the same bounded search budget. Selection was frozen before
these curve scores. The failed earlier sandwich-normalization screen is
preserved [separately](../recurrent_screen_20261023/REPORT.md); no 1B fit from
that screen launched.

| Endpoint domain | Ordinary mean NLL | Recurrent mean NLL |
| --- | ---: | ---: |
| web | 3.000095 | 3.037707 |
| math | 2.262162 | 2.281532 |
| code | 1.337085 | 1.357587 |

The primary weights are web/math/code 0.6/0.2/0.2. Every scored source token is
retained; domains use isolated documents, including short documents and tails.
The curve panel contains 614 documents, 777,348 real targets and 360 source groups.

At matched estimated training matmul FLOPs, recurrence is also worse:
ordinary **2.519907** at 1,004,887,590 targets versus
recurrent **2.695650** at 593,774,218 targets; gain
**-0.17574 [-0.17845, -0.17303]**.
Both endpoints are independently annealed. The recurrent budget is 0.0286%
smaller because updates are indivisible. At equal tokens, recurrence costs
**1.6919×** the estimated dense training matmul FLOPs. These estimates include
the vocabulary projection; they exclude elementwise work, optimization and
compilation, and are not hardware performance counters.

A separately implemented FP32 CPU model independently scores all six terminal
checkpoints on the complete panel. Its maximum per-model difference from
native BF16 NLL is **0.0000326 nats**; all three paired deficits remain about
0.0305 nats. The independent implementation uses separate Q/K/V projections
and a separately implemented RoPE/attention path, and passes forward plus full
gradient agreement. This is numerical verification on the same data, not another
held-out replication. [CPU audit](independent-cpu-verification.json).

All native phases pass a CPU loss/full-gradient probe before updates, full
checkpoint recovery and exact equality of model plus optimizer state across
64 replicas. Qualification measures **604,416 real targets/sec** over 100 warm
recurrent updates on all 16 chips. Actual terminal phases measure 590k–641k.
A brief runtime sample has all 64 logical cores active and no execution errors;
its roughly 65% busy occupancy is not peak-FLOP utilization or a campaign average.

The receipt ledger includes 70 closed native/CPU invocations, including the
abandoned normalization screen, tuning and readouts: **47.47 assigned chip-hours**,
with **3.54 hours** in the union of accelerator-active intervals. CPU data
preparation, CPU probes and idle reservation time are separate; reused ordinary
training is not charged again. [Execution receipts](execution-costs.json).

The primary controller completed at **07:05:38 UTC on October 23**. The planned
follow-on qualification, 3B continuation, same-weight inference-unroll matrix
and TEST confirmation **did not run**. The orchestration did not advance after
the completed primary, leaving accelerators idle afterward. The frozen 11:12
campaign deadline passed. A 3.25B-target append-only natural corpus is prepared,
with unchanged original packed prefix and DEV/TEST, but it is not a training result.
TEST remains sealed. Experimental fast-path scripts are unqualified drafts.

These results reject a useful gain from this fixed two-pass pre-norm recipe at
1B targets. They do not establish that recurrence cannot help after longer
training, with a different stable architecture, or on downstream tasks. Three
paired seeds quantify initialization variation conditional on one frozen corpus
and data order. Source-group bootstrap intervals, reported separately, condition
on fitted checkpoints and do not replace seed replication.

[Machine-readable results](summary.json) · [Per-seed CSV](learning-curve.csv) ·
[Frozen protocol](../../docs/PRELN_RECURRENT_PROTOCOL.md) ·
[Analysis source](../../scripts/analyze_preln_learning.py).
