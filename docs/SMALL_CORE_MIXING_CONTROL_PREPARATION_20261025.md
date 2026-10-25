# A stronger cheap control — CPU preparation only

The completed equal-data discovery combines repeated input/state mixing with
additional transformer-core executions. The original one-pass family executes
both once. A trained control separating those changes is still owed.

Prepare a parameter-exact control that repeats only the learned adapter,then
executes the core once,followed by the unchanged output path. Keep the original
depth/noise assignments and differentiate every mixing application. Place the
single core after the mixing sequence so it can process the resulting state;
at one mixing step the entire function and gradients match the existing family.
No algebraic projection hoisting or runtime skip is credited in its cost model.

This asks whether inexpensive mixing and initial-state attenuation can recover
the source-efficiency gain without extra attention/MLP executions. It does not
assume that they do. If the control wins,keep the stronger baseline and do not
attribute its gain to recurrent transformer computation. An inference ablation
or an unqualified fixed learning rate would not substitute for training it.

This preparation adds CPU operator/gradient/causality and work-ledger checks
only. It does not adopt a checkpoint,launch hardware qualification,register a
scientific fit,extend the current campaign or open a held-out panel. Any later
training needs a new bounded protocol,fair LR selection,exact work accounting,
native update/recovery/throughput checks and separately cooled comparisons.
Finish the current work controls and prefill measurement first.
