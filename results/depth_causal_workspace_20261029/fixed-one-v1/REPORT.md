# Fixed-depth training recovers quality, but does not beat ordinary

The schedule-only comparison is complete. Fixed one-step training improves
0.073799 nats over the supported one-step exit of mixed zero/one/two training,
with conditional source-group 95% interval [0.069108, 0.078267]. This passes the
registered 0.02 schedule-effect screen. It still loses 0.040875 to the strongest
tied ordinary model, interval [−0.046910, −0.035068] for its gain. The ordinary
competitiveness gate fails; the joint advancement decision is negative.

| Predictor | Exact development NLL ↓ | Web | Math | Code |
|---|---:|---:|---:|---:|
| Strong tied ordinary | 3.287781 | 3.789349 | 2.957114 | 2.113745 |
| Fixed one-step, exact depth-causal | 3.328656 | 3.823924 | 3.001760 | 2.169748 |
| Mixed training, supported one-step exit | 3.402455 | 3.894405 | 3.073435 | 2.255623 |

Every domain shows the schedule improvement and the remaining conventional
shortfall. The earlier persistent fixed-one model also remains better by
0.065922, interval [0.061129, 0.070669] for its advantage. The exact depth-causal
model improves execution cost but changes which earlier latent states later
predictions can access; this comparison does not establish a universal
advantage of either memory graph.

The three fresh fits preserve the original seed 9901 initializer, source order,
128,025,288 authentic targets, uniform next-token objective and full 1,957-update
schedules. Learning rates 0.0003, 0.001 and 0.003 receive identical opportunities;
0.001 is the interior optimum. Only the trained one-step exit is evaluated.
Zero- or two-step performance of this fixed-depth model is unsupported.

All three mature full-global gradient, CPU state-restoration and 64-replica
checks pass. Maximum gradient-family relative errors are 0.00002115,
0.00003960 and 0.00004790 against the 0.02 limit. Independent FP64 readout covers
all 384 reused development sources. Eighteen cache/full-mask/token-major checks
agree within 2.14e-14 in log probability. Independent loss-sum aggregation
reproduces the comparison point estimates within 1e-12.

A retained fit uses 49,069,855,327,911,936 nominal training matrix FLOPs,
1.12 times strongest ordinary at equal data and 0.984026 times mixed training.
The complete campaign, including three fits and Native evaluation, consumes
6.553761 assigned chip-hours. Its 7.499198 whole-instance opportunity chip-hours
include CPU and controller time and must not be added again. Prior disposable
qualification costs remain separately reported.

This establishes a substantial schedule penalty within this selected seed and
recipe. It does not identify the optimization mechanism, establish iterative
reasoning, or justify scaling. The earlier positive TRAIN gradient alignment
does not rule out a learning tradeoff across budgets. All earlier failed
multi-budget gates remain failed. Both fresh reserves remain unscored and
terminal alpha remains zero.

[Registration](../../../docs/FIXED_DEPTH_CAUSAL_COMPARISON_20261029.md) ·
[All comparisons and decisions](compare.json) · [LR selection](select.json) ·
[Exact readout](readout.json) · [Independent aggregation](independent-aggregate-check.json) ·
[Execution ledger](execution-costs.json).
