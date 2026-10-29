# Complete-fit and graph-first-use timing

Post-hoc accounting of all six fixed replication fits. Remove only the first occurrence of each compiled numerical-depth graph from remaining-update summaries, never timing outliers. First use includes compilation plus an actual update, not isolated compile time. Complete assigned Native attempts retain startup, checkpointing and audits. Cross-launch descriptive timing, not controlled hardware-time-matched predictive efficiency. No new quality score, fresh reserve or endpoint change.

| Seed | Model | Complete Native seconds | First graph uses, seconds | Remaining update mean, seconds | Assigned chip-hours |
| --- | --- | ---: | --- | ---: | ---: |
| 730201 | latent | 774.403 | u1: 105.182, u5: 87.362 | 0.262712 | 3.4418 |
| 730201 | ordinary | 692.804 | u1: 57.061 | 0.294764 | 3.0791 |
| 730201 | pause | 460.529 | u1: 45.580 | 0.179892 | 2.0468 |
| 730237 | latent | 788.791 | u1: 106.483, u5: 85.225 | 0.268358 | 3.5057 |
| 730237 | ordinary | 682.689 | u1: 55.362 | 0.291084 | 3.0342 |
| 730237 | pause | 458.734 | u1: 46.335 | 0.180135 | 2.0388 |

Both latent fits have faster remaining updates than their tied ordinary controls, but slower complete Native fit attempts. The extra first graph use materially affects these short runs. It is not valid to remove that cost from the completed comparison or extrapolate a hardware-time quality win.

The frozen trainer's existing `mean_warm_update_seconds` excludes only the first three updates. For mixed-depth training it therefore includes the second graph's first use at update5. This separate report preserves that original field and identifies graph first use explicitly.

All six full endpoints and numerical audits are retained. Numerical refinement rounds do not measure inference thought length.
