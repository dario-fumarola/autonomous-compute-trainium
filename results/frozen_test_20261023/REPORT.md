# Frozen TEST confirmation

All model identities, trained depths, evaluator and analysis sources were
frozen before opening TEST. This confirms the original three-seed comparison;
the untied replications and resource control were added before TEST access.

| Model | Seed | Endpoint | TEST weighted NLL |
| --- | ---: | --- | ---: |
| ordinary | 23 | 1B | 2.575164 |
| shared | 23 | 1B | 2.608864 |
| untied | 23 | 1B | 2.550684 |
| ordinary | 47 | 1B | 2.576956 |
| shared | 47 | 1B | 2.609372 |
| untied | 47 | 1B | 2.555731 |
| ordinary | 71 | 1B | 2.584139 |
| shared | 71 | 1B | 2.612684 |
| untied | 71 | 1B | 2.543315 |
| ordinary | 23 | 3B | 2.391002 |
| shared | 23 | 3B | 2.393539 |
| ordinary | 23 | matched FLOPs | 2.478950 |

Original three-seed ordinary-minus-shared gain: **-0.031554 [-0.038220, -0.024888]**. Decision at the declared 0.01-nat threshold: `USEFUL_GAIN_RULED_OUT_AT_THIS_EXPOSURE`.

| Paired comparison | Mean gain | 95% training-seed interval |
| --- | ---: | ---: |
| ordinary_to_shared_three_seeds | -0.031554 | [-0.038220, -0.024888] |
| ordinary_to_untied_three_seeds | 0.028842 | [0.002752, 0.054932] |
| ordinary_to_untied_fresh_seeds | 0.031024 | [-0.093489, 0.155537] |
| shared_to_untied_three_seeds | 0.060396 | [0.040287, 0.080506] |
| shared_to_untied_fresh_seeds | 0.061505 | [-0.038419, 0.161429] |

| Single-pair comparison | Gain | 95% source-group interval, conditional on weights |
| --- | ---: | ---: |
| ordinary_to_shared_3b | -0.002538 | [-0.005603, 0.000730] |
| ordinary_matched_flops_to_untied_1b | -0.071734 | [-0.077985, -0.065560] |

Positive gains favor the candidate. Source-group intervals do not replace
independent training seeds. Combined untied intervals include the discovery
seed; fresh-seed intervals have df=1. The larger untied model is not parameter
matched. The ordinary resource control matches estimated dense training matmul
work, not every cost. All domain scores are in the machine-readable
report. The original DEV outcome and protocols remain preserved; TEST does
not replace them. No tuning, new model variant, checkpoint selection, or
exploratory unroll uses this panel. Further scientific variants need a
separate held-out panel.

[Machine-readable confirmation](summary.json) · [Frozen release](release.json) ·
[Protocol](../../docs/FROZEN_TEST_CONFIRMATION.md).
