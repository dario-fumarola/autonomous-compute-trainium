# Fixed-two mixing control: tuning complete

The control performs exactly two adapter/input mixes,then one transformer-core
execution. It retains the same parameter inventory and original NTP objective.
It tests whether a cheaper one-pass parameterization can explain the small
matched-work benefit of fixed-core recurrence. The older mixing-only study
trained varying depths and cannot replace this fixed-depth control.

The CPU-only preparation passes forced-depth/work-table readback and original
initializer/source checks. Nominal work is277,003,247,086,080 matrix operations
per512-window update,including the common zero-weight head and32 replicated
optimizer updates. Preparation takes1.77 CPU seconds and performs zero model
forwards,optimizer updates or accelerator work.

Three exact learning-rate opportunities are frozen:0.008,0.016,0.032,seed4127,
301,465,908 real targets per trial with complete cooldowns. Every counted121-
update prefix needs independent complete512-window gradients,state recovery,
replica agreement and100 measured warm updates before continuing. All three
cooled scores and the selected final gradient gate are required. The original
qualified finite controller and mixing-model implementations are reused.

The conservative initial estimate is4307.83 seconds under a5400-second cap;
actual prefix timings must reconfirm the remaining budget.281 CPU tests pass.
All four training/capture/CPU-evaluation module routes pass a dispatcher smoke
check;that check is not numerical or hardware qualification.

The predecessor curve and prefill are complete. Native TUNE was admitted under
its own original90-minute clock. All three counted121-update prefixes finish
with complete replica states equal;all three independent global-gradient gates
pass. Actual100-update times are107.640,107.548 and103.616 seconds for the three
rates. The remaining conservative bound is3080.00 seconds against4341.90 seconds
remaining on the original clock,so all three continue to their cooled endpoints.
All three cooled trials and the selected endpoint's independent 512-window
gradient audit pass. Their TUNE NLLs are 3.003628408, 2.959936488 and 2.946774035,
respectively. The selected rate is 0.032, at the upper grid boundary; the grid
was not extended. This selection score does not establish the longer control's
quality. The stage made 1,728 counted training updates and 904,397,724 source
target presentations. Its 10 native phases used 6.421885 assigned chip-hours;
elapsed whole-instance opportunity cost was separately 13.117270 chip-hours,
including independent CPU tails. These costs are not additive.
This registration admits no full scientific control fit or confirmation run.
No TEST,reserve groups or terminal alpha are accessed.

[Protocol](../../../docs/FIXED_TWO_MIXING_CONTROL_20261026.md) ·
[CPU preparation](preparation.json) · [Frozen queue](queue-plan.json) ·
[CPU tests](cpu-tests.json) · [Module routing](module-routing-smoke.json).
