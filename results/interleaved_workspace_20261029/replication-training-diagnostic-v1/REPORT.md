# Replication variation is also visible in training

This is a post hoc summary of the six saved complete training traces. It runs
no optimizer updates, scores no new held-out example and changes no gate.
All source receipts are hashed in[report.json](report.json).

Positive gaps favor generated latent state. The numbers below weight each
update by its actual natural-target count.

| Seed | Last stable phase, over tied ordinary | Cooldown, over tied ordinary | Cooldown, over trained pauses |
|---|---:|---:|---:|
|730201|0.002924|0.005570|0.133022|
|730237|0.033025|0.033692|0.135799|

The stable interval is updates1500–1761 and cooldown1762–1957. These intervals
were chosen after the completed runs and are descriptive. Training losses use
the registered mixed numerical solver depths; they are not additional exact
sequential development checkpoints.

The ordering resembles the completed exact development results: the first
seed's conventional margin is small, while the second's is larger. This makes
evaluation-panel noise alone an inadequate explanation of the observed
variation. It does not identify an architectural cause. Both model families
contribute to the variation, and both seeds remain in every reported result.

Gradient clipping is uncommon after warm-up: at most one clipped update in
either latent cooldown, versus none in either ordinary cooldown. The trace
does not show a persistent clipping problem at the endpoint. This is not a
substitute for the already completed numerical audits and does not prove that
the recipe is optimally tuned.

The active supported-budget experiment is unchanged. The fixed-one useful
replication gate remains failed; no failed seed is discarded or extended.
