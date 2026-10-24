# CPU optimizer qualification

Three CPU execution modes pass against the pinned released tensor algorithm:
plain/eager, fused/eager, fused/AOT-eager. Each checks five prescribed gradient
updates, varying learning rate, momentum and cautious decay. Maximum accumulated
parameter error relative to a reference step is6.88e-6; Muon moments match exactly.
Full nested state and the next update match exactly after serialization.

Final attempt:4.465s including imports, zero native work. These are numerical
software fixtures, not model learning. The earlier narrower attempt is retained.

[Final evidence](attempt2-state-and-aot/report.json) ·
[Receipt](attempt2-state-and-aot/execution-costs.json) ·
[Scope and remaining gates](../../docs/PARCAE_OPTIMIZER_AND_STAGES_20261024.md).
