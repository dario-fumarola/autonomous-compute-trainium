# A control for persistent generated memory

A CPU reference now generates a state for each natural-token prediction while
preventing all later positions from reading that state. Parameters and current
prediction depth are unchanged. This can test whether the persistent model's
benefit requires carrying generated content forward, once both are trained.

Sixteen focused tests pass. The complete suite passes **765 tests, two skips**.
Independent prefix recomputation, parallel masked attention and source-KV reuse
agree in outputs, full gradients and optimizer steps. All current source
code remains unchanged for the running replication.

At the existing 51.75M shape and 512 source positions, the source-reuse graph
counts 195,890,774,016 training matmul FLOPs per window, including full backward.
That is 1.65617 times the original ordinary graph, and 28.38% below the
recomputing control. These are dispatched operation counts, not measured
hardware time. Reuse does not detach source-state gradients.

No model has been scientifically trained or evaluated under this control.
The prepared 45-minute hardware qualification waits for all six replication
fits to pass their audits; it has no automatic scientific-training promotion.

[Exact CPU receipt](report.json) ·
[Qualification design](../../../docs/EPHEMERAL_WORKSPACE_QUALIFICATION_20261029.md).
