# Exploratory position breakdown of the negative sparse result

This analysis follows the completed quality comparison. The attempted prospective draft was rejected because quality artifacts already existed; that refusal is preserved. These are exploratory, unadjusted conditional source-group intervals on reused DEV, not a new advancement criterion. No model was fitted, reselected or evaluated on a fresh reserve.

Every broad and domain NLL reconstructs the completed exact comparison within 1e-8. All five bins and all three pairwise comparisons are reported. Positive gain favors the first named model.

## generated over ordinary

| Position bin | Targets per view | Gain per target | Conditional 95% interval | Contribution to broad gain |
|---|---:|---:|---|---:|
| before_first_slot | 127 | -0.004077 | [-0.011850, +0.004231] | -0.001011385 |
| slot_prediction | 4 | -0.106686 | [-0.158068, -0.056963] | -0.000833485 |
| after_1_to_8 | 24 | -0.021763 | [-0.037765, -0.005925] | -0.001020161 |
| after_9_to_32 | 72 | -0.001926 | [-0.012974, +0.009111] | -0.000270851 |
| after_33_to_127 | 285 | -0.002563 | [-0.007690, +0.002948] | -0.001426471 |

| Position bin | Web gain | Math gain | Code gain |
|---|---:|---:|---:|
| before_first_slot | -0.002764 | -0.001873 | -0.010223 |
| slot_prediction | -0.117897 | -0.096876 | -0.082862 |
| after_1_to_8 | -0.033239 | -0.020235 | +0.011136 |
| after_9_to_32 | -0.000495 | -0.006858 | -0.001286 |
| after_33_to_127 | -0.003788 | -0.003850 | +0.002400 |

## generated over constant

| Position bin | Targets per view | Gain per target | Conditional 95% interval | Contribution to broad gain |
|---|---:|---:|---|---:|
| before_first_slot | 127 | +0.002047 | [-0.005483, +0.009778] | +0.000507855 |
| slot_prediction | 4 | +0.085412 | [+0.030083, +0.142921] | +0.000667281 |
| after_1_to_8 | 24 | -0.014708 | [-0.031212, +0.002610] | -0.000689417 |
| after_9_to_32 | 72 | +0.001798 | [-0.008520, +0.012629] | +0.000252777 |
| after_33_to_127 | 285 | -0.000175 | [-0.005346, +0.005332] | -0.000097293 |

| Position bin | Web gain | Math gain | Code gain |
|---|---:|---:|---:|
| before_first_slot | +0.004986 | -0.000541 | -0.004182 |
| slot_prediction | +0.102183 | +0.113933 | +0.006580 |
| after_1_to_8 | -0.015994 | -0.010997 | -0.014560 |
| after_9_to_32 | +0.004431 | -0.007785 | +0.003480 |
| after_33_to_127 | +0.001392 | -0.004749 | -0.000301 |

## constant over ordinary

| Position bin | Targets per view | Gain per target | Conditional 95% interval | Contribution to broad gain |
|---|---:|---:|---|---:|
| before_first_slot | 127 | -0.006125 | [-0.013328, +0.001008] | -0.001519240 |
| slot_prediction | 4 | -0.192098 | [-0.251446, -0.134220] | -0.001500766 |
| after_1_to_8 | 24 | -0.007056 | [-0.023594, +0.009739] | -0.000330744 |
| after_9_to_32 | 72 | -0.003724 | [-0.014322, +0.006722] | -0.000523627 |
| after_33_to_127 | 285 | -0.002388 | [-0.007451, +0.002966] | -0.001329178 |

| Position bin | Web gain | Math gain | Code gain |
|---|---:|---:|---:|
| before_first_slot | -0.007750 | -0.001333 | -0.006041 |
| slot_prediction | -0.220080 | -0.210809 | -0.089442 |
| after_1_to_8 | -0.017246 | -0.009238 | +0.025696 |
| after_9_to_32 | -0.004926 | +0.000927 | -0.004766 |
| after_33_to_127 | -0.005180 | +0.000899 | +0.002701 |

## Interpretation and limits

The largest per-target penalty occurs at predictions forced through the extra slot. Generated feedback improves these boundary predictions relative to the constant-input workspace, but still loses to ordinary. The next eight positions also lose to ordinary; the generated-versus-constant gain at the boundary is approximately canceled by its deficit in those following positions.

The four boundary predictions and next-eight bins together account for 40.63% of the generated model’s broad deficit. The remaining deficit is outside these bins. Some difference appears before any slot, where changed backbone training can matter. This does not identify a causal memory benefit, and a favorable code bin is not a new positive result. No memory corruption or unsupported thought budget was evaluated.

The four-worker CPU analysis took 88.803 wall-seconds, zero chips and zero optimizer updates. Both fresh reserves remain sealed; terminal alpha is zero. [All aggregates](report.json) · [Exploratory design](../../../docs/CHUNKED_COMPLETION_POSITION_EXPLORATORY_20261029.md).
