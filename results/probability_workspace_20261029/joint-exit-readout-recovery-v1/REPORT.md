# Joint-exit objective: complete FP64 readout

One new 79.18M model, 200,190,896 natural targets, complete cooldown and passing mature training audits. All six endpoints are recomputed in FP64 using the same source targets. Lower NLL is better.

| Predictor | Broad NLL | Web | Math | Code |
|---|---:|---:|---:|---:|
| Ordinary, matched matrix work | 2.980724 | 3.459720 | 2.685691 | 1.838766 |
| Ordinary, equal data | 3.253961 | 3.750421 | 2.925745 | 2.092796 |
| Final-only workspace | 3.270157 | 3.764671 | 2.945298 | 2.111476 |
| Joint, shallow exit | 3.271351 | 3.766902 | 2.940157 | 2.115890 |
| Joint, refined exit | 3.272421 | 3.764222 | 2.944078 | 2.125361 |
| Joint, permuted values | 3.588184 | 3.952801 | 3.216175 | 2.866344 |

Positive differences favor the joint/refined model as named. Intervals are conditional on these weights and the reused DEV source groups; they do not measure training-seed uncertainty.

| Comparison | Gain | Conditional 95% interval |
|---|---:|---:|
| supported_refinement | -0.001071 | [-0.002689, +0.000532] |
| ordinary_equal_data | -0.018460 | [-0.023319, -0.013423] |
| ordinary_equal_work | -0.291698 | [-0.300013, -0.283158] |
| final_only_objective | -0.002264 | [-0.007173, +0.002405] |
| shallow_vs_ordinary_data | -0.017390 | [-0.022063, -0.012471] |
| shallow_vs_ordinary_work | -0.290627 | [-0.298780, -0.282421] |
| proposal_content_intervention | +0.315763 | [+0.302245, +0.330271] |

Registered development gates: `{'supported_refinement': False, 'ordinary_equal_data': False, 'ordinary_equal_work': False}`.

Every real target has total weight one: half on each supported exit. Both exits receive ordinary NTP, and gradients remain connected through the generated proposal. Initialization, parameters, source order, LR and endpoint match the historical final-only candidate. Matrix work is unchanged.

The original Native evaluation stopped on the permuted-value condition: maximum target error 0.000222091 versus the 0.000200 limit. Its partial scores are not used. No tolerance was relaxed and no checkpoint was changed. The CPU recovery recomputes the candidate and every reused control in FP64 within the original campaign deadline.

This single-seed, early-learning comparison uses one common LR and a repeatedly inspected panel. A better shallow-to-deep difference cannot replace an absolute ordinary-model advantage. Permutation damage alone is not reasoning evidence. All authored-answer and domain results remain in the complete analysis. Both fresh source reserves remain unscored.

The new fit used 4.573590 assigned chip-hours; the failed Native readout used 0.727966. CPU recovery has zero assigned accelerator time. Qualification, historical controls and instance-opportunity costs remain in their separate ledgers and must not be double-counted.

[Complete analysis](report.json) · [Recovery costs](execution-costs.json) · [Frozen recovery](plan.json) · [Original stopped history](../joint-exit-screen-v1/report.json)
