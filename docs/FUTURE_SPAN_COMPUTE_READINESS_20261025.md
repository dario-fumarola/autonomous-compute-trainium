# Work-matched continuation feasibility

This pre-outcome CPU audit reads the existing source counters and execution
ledgers, without reading MAIN quality scores. It reconstructs the nominal
matrix work from the shape-only model, attention, vocabulary/future heads,
full gradients and every replica's Muon updates. No weights are initialized,
model forwards performed, checkpoint adopted or training run admitted.

The recurrent1.005B endpoint costs650,864,708,520,837,120 matrix operations.
The largest complete one-pass endpoint within that budget is2365 updates and
1,237,788,282 real targets. It costs650,724,597,171,601,920 operations,0.021527%
less;another complete update would exceed the budget. Both one-pass objective
cells have this geometry. The paid training-only head remains in both controls,
including the zero-weight NTP control. This is not a head-free ordinary recipe
or a wall-time/hardware-cost match.

A valid continuation can branch from the actual stable960-update/502,444,236-
target state. Its own cooldown starts at1182 updates/618,634,348 targets and
ends at2365 updates. The full inherited prefix remains charged to the model;
only incremental branch work is a separate campaign cost. Source and auxiliary
ledgers already cover this endpoint without wrapping or fetching more text.

This establishes feasible geometry, not launch readiness or model quality.
Cross-plan provenance transport and exact full-state/RNG recovery still need
qualification before registering any continuation. Original stable checkpoints
must remain intact. The current MAIN source, schedule and decision rules do
not change. Stronger ordinary/cheap controls, inference costs and independent
fresh confirmation remain required even if a discovery candidate passes.

[Measured ledger audit](../results/future_span_compute_readiness_20261025/report.json)
uses no chips and reads no held-out scores. All218 CPU tests pass;the original
119 frozen MAIN source files and the three new analysis files are unchanged.
