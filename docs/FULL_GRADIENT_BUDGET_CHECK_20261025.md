# Budget check after the completed two-seed discovery

This is research and metadata feasibility,not an admitted experiment. The
current work controls and queued prefill qualification keep their original
source freezes,deadlines and priority. No new model or source stream is created.

The primary [full-gradient scaling study](https://arxiv.org/html/2610.26030v2)
reports fixed-size advantages from two fully differentiated loops at sufficiently
large budgets. Its measured points consistently favor them beyond roughly30
ordinary-equivalent tokens per non-embedding parameter,with about0.01 nats at50.
Its free-size optimum still favors ordinary models. Those thresholds belong to
its architecture,data and optimization;our value embeddings,core fraction and
sampled depths prevent directly transferring them.

The [Parcae paper](https://arxiv.org/html/2604.12946v1) includes a140M isoFLOP
suite beginning at1e18 operations and selects mean recurrence two at its lowest
budget. Its loss frontier improves,but low-budget downstream CORE is worse
than its fixed-depth reference (7.6 versus7.9). Its accounting includes output
and attention but differs from our complete ledger,which also charges the
auxiliary head and every replica's optimizer matrices. A nominal1e18 comparison
here would not be a faithful reproduction or a guaranteed crossover.

## What is feasible without changing data identities

The original independent audit already reconstructed all4,097 work rows. A new
read-only calculation rechecks their hashes and the source-count prefix. Under
the fixed1e18 ceiling,the largest complete recurrent endpoint and its exact
work comparator are:

| Potential endpoint | Updates | Real targets | Complete nominal matrix work | Cooldown starts |
| --- | ---: | ---: | ---: | ---: |
| Recurrent NTP | 2,949 | 1,543,437,724 | 999,770,245,750,559,232 | 1,474 |
| One-pass matched work | 3,633 | 1,901,424,050 | 999,612,034,471,217,664 | 1,816 |

The comparator is0.015825% below the recurrent endpoint. A separately cooled
equal-data one-pass endpoint also fits. All following-gradient rows fit the
existing counts/depth/work files. Original stable update960 is early enough
for every new cooldown. Thus no data/depth-signature extension is needed.
No future checkpoint is claimed to exist,and no trained state was adopted.
[Exact geometry and provenance](../results/future_span_later_budget_readiness_20261025/report.json).

This budget was selected after MAIN discovery and primary-paper inspection,
before any current matched-work score. The audit reads metadata only and
cannot authorize training. Any later run needs its own frozen protocol,budget,
stable-state transport/native checks and complete cooled comparators.

## The stronger cheap control remains important

Our current recurrence repeats both input mixing and attention/MLP blocks.
The parameter-exact one-pass control applies that mixing only once. My inference
from this code structure is that a trained control repeating only the cheap
mixing,with one core execution and the same depth/noise assignments,could more
directly identify what the extra block executions contribute. An inference
ablation would not substitute for training that control.

If the cheap control recovers the benefit,retain it as a stronger baseline and
do not credit expensive recurrence. It also needs prospective qualification and
fair tuning. Finish the current resource evidence before choosing between that
control and a larger exposure;do not automatically increase training until a
positive appears. Strong ordinary controls and fresh confirmation remain owed.
