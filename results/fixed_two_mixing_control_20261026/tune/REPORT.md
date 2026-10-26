# Fixed-two mixing control: counted prefixes complete, audits running

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
with complete replica states equal;their independent global-gradient gates
are running. No cooled selection outcome is available yet. After the prefix
audits,the actual100-update timings must reconfirm remaining feasibility.
This registration admits no full scientific control fit or confirmation run.
No TEST,reserve groups or terminal alpha are accessed.

[Protocol](../../../docs/FIXED_TWO_MIXING_CONTROL_20261026.md) ·
[CPU preparation](preparation.json) · [Frozen queue](queue-plan.json) ·
[CPU tests](cpu-tests.json) · [Module routing](module-routing-smoke.json).
