# Exact depth-causal actual-shape qualification passes

All64ranks/all16chips pass the registered zero/one/two-budget cold loss and
full-gradient comparisons.104disposable updates complete;100warm updates
average0.177188seconds. Budget0/1/2means are0.156502/0.177368/0.198320seconds.
These are measured execution times on disposable TRAIN views, not quality.

All replicas agree in full state. A fresh process restores update103 and exactly
reproduces update104including model, optimizer and RNG. The mature full-global
128-window gradient agrees with independent CPU full-mask recomputation: maximum
family relative error1.28685693e-06, loss error1.50501728e-06nats.
CPU audit duration52.329seconds. No tolerance was relaxed.

Complete warm/replay Native attempts cost2.032899assigned chip-hours,
including compilation, checkpointing and restart. No scientific fit, held-out
score, fresh reserve read or qualified-state reuse occurred. Scientific fits
require their own immutable plan, original random initialization and controls.

[Full numerical/state evidence](report.json) · [Costs](execution-costs.json) ·
[Frozen qualification](plan.json) · [CPU reference](../cpu-v1/REPORT.md).
