# Prospective mixed numerical-round qualification

No scientific fitting is admitted here. Finish the active fixed-two tuning
queue intact. The learned-state consistency diagnostic separately determines
whether its learned graph agrees with sequential generation. This preparation
does not use tuning quality to change the completed fixed-round experiment.

PonderLM-2 Appendix E motivates testing exposure to both two and three Jacobi
rounds. The rounds approximate one persistent latent token per source position;
they are not different thought-token budgets. The first step is a disposable
actual-shape qualification with both compiled graphs resident, because its
memory and execution behavior cannot be assumed from a single-graph run.

Reuse the qualified grouped-storage model at51,747,328 parameters,512 source
positions,FP32,64 ranks and two windows per rank. Use the existing disposable
natural-TRAIN qualification views, fresh seed9841, AdamW learning rate0.0003,
and alternating K2/K3 updates. Cold independent CPU gradients must pass for
both graphs. Three initial plus100 warm updates measure50 warm updates at each
depth. Update104 usesK3, saves its full global gradient, and must replay exactly
in a fresh process from the complete update103 model/optimizer/RNG state.
Independent CPU audit covers every target in the128-window global batch.

Retain the existing2% per-family gradient and0.003 loss gates, all-replica/full-
state equality, no unexpected fallbacks, and0.8-second warm-update ceiling.
Report cost by numerical depth and all compilation/checkpoint/audit costs.
The Native192GiB worker cap,512GiB host reserve,35GiB disk floor,10,000 inode
reserve and four-GiB artifact allowance remain. Two snapshot projections must
pass before registration. Bound the queue to90minutes, Native phases30minutes
and CPU audit20minutes. No overlap with the active scientific Native owner.

Do not retain any qualification weights for science. Passing only establishes
that mixed-round fitting can be implemented correctly and at measured cost.
A later scientific protocol must independently specify the random depth
schedule, equal LR opportunities, complete target/work budgets, controls,
cooldowns and unchanged sequential-consistency diagnostic. It must preserve
the fixed-two results regardless of their direction. No automatic scaling or
confirmation follows from this numerical test.
