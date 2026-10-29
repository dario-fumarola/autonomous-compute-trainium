# Exact depth-causal multi-budget learning is negative

The complete registered three-LR screen fails all four advancement gates.
Executing the revised graph exactly and cheaply does not make its second
latent step useful at this endpoint. No fit is extended or promoted.

| Supported budget | Exact development NLL | Web | Math | Code |
| --- | ---: | ---: | ---: | ---: |
| 0 | 3.473785 | 3.966631 | 3.141940 | 2.327094 |
| 1 | 3.402455 | 3.894405 | 3.073435 | 2.255623 |
| 2 | 3.408117 | 3.900235 | 3.078748 | 2.261129 |

LR 0.001 is the interior winner selected by two-step NLL. Every arm uses
1,957 updates, 128,025,288 authentic targets and the original per-update
source/budget assignment. There is no target reweighting, router, new parameter
or generated training corpus. All three endpoints pass full mature gradients,
state restoration and 64-replica checks before scoring.

| Comparison: control minus candidate NLL | Gain | Conditional 95% interval | Gate |
| --- | ---: | --- | --- |
| Two over supported one | -0.005662 | [-0.006818, -0.004510] | Fails |
| Two over independently trained fixed one | -0.145383 | [-0.151338, -0.139486] | Fails |
| Two over strongest tied ordinary | -0.120335 | [-0.127000, -0.113534] | Fails |
| Zero over same-shape ordinary | -0.096674 | [-0.101202, -0.092011] | Fails |

The second step hurts in web, mathematics and code. The zero-step path is
0.096674 nats worse than its independently trained ordinary counterpart, far
outside the 0.005 point guard. The positive within-model one-over-zero gap
therefore cannot be treated as progress over ordinary training.

The same trained checkpoint is evaluated at all its supported budgets. The
independent fixed-one control uses the earlier persistent-memory graph, so
that comparison does not isolate fixed-versus-mixed-budget training in the
new graph. A separately trained fixed-one version of the exact depth-causal
graph is an unfilled comparison, not an outcome established here.

## Numerical and resource evidence

All 48 FP64 cached/full-mask/token-major checks pass; maximum log-probability
error is 2.49e-14. Complete source identities and target counts agree across
all comparisons, and independent reconstruction from saved loss sums agrees
to 1e-12. These are independent numerical/aggregation checks, not independent
training replication. All intervals condition on one selected training seed
and the repeatedly inspected 384-source development panel.

Each fit costs 49,866,412,847,529,984 nominal training-matrix FLOPs, 1.138181
times strongest tied ordinary at equal source exposure. Complete fitting,
launching and common Native evaluation consume 8.679036 assigned chip-hours.
The controller's 10.851517 opportunity chip-hours include CPU
phases and must not be added to assigned work. Prior disposable qualification
costs 2.032899 assigned chip-hours separately. Exact CPU readout takes
80.02 seconds, with zero accelerator assignment.

A separate descriptive analysis reuses the existing TRAIN gradients. Their
whole-model directions are aligned across budgets at all three endpoints;
this does not support broad opposing gradients at those states, but does not
identify a causal explanation or rule out optimization interference elsewhere.
[Gradient analysis](../gradient-alignment-v1/REPORT.md).

Both fresh reserves remain unscored and terminal alpha remains zero. These
findings do not establish a universal limitation of latent computation, and
provide no efficiency, learned-allocation or reasoning-transfer claim.

[Protocol](../../../docs/DEPTH_CAUSAL_LEARNING_SCREEN_20261029.md) ·
[All scores and gates](compare.json) · [Learning-rate results](select.json) ·
[Independent aggregation](independent-aggregate-check.json) ·
[Complete execution costs](execution-costs.json).
