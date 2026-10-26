# Completed fixed-two development curve

All six cooled endpoints and independent full512-window gradient audits are complete. All models start from the same original seed5231 random initializer,retain141,963,840 deployed parameters and train only ordinary NTP. This is one reused development seed on consumed DEV.

| Model | Real training targets | Independent FP32 DEV NLL |
| --- | ---: | ---: |
| One,1B | 1,004,887,590 | 2.614669468 |
| Fixed2,1B | 1,004,887,590 | 2.579990614 |
| One,1B work control | 1,210,049,128 | 2.584245404 |
| One,2B | 2,009,762,680 | 2.516719486 |
| Fixed2,2B | 2,009,762,680 | 2.484647907 |
| One,2B work control | 2,420,609,556 | 2.496663431 |

| Comparison | Fixed2 gain,nats | Conditional source-group95% interval |
| --- | ---: | ---: |
| Equal data,1B | +0.034678854 | [0.032680362,0.036699020] |
| Matched work,1B | +0.004254790 | [0.002268938,0.006303373] |
| Equal data,2B | +0.032071579 | [0.029791000,0.034485709] |
| Matched work,2B — primary | +0.012015524 | [0.009628350,0.014405026] |

The1B matched-work gain is below the0.01 useful-effect threshold. The primary2B point exceeds it,but its interval overlaps it. These intervals resample518 source groups conditional on fitted models;none estimates training-seed variation or provides independent confirmation. Web,math and code all favor fixed2 in the primary pair.

The equal-data gain changes little between these two exposures. Extra One training improves NLL by0.030424 at1B and0.020056 at2B;the smaller alternative gain leaves more of the fixed-depth benefit at2B. This is arithmetic from two completed comparisons,not a scaling-law fit or a reason for an automatic larger run.

Both nominal work matches count the common zero-weight auxiliary head and every replica's optimizer matrices. The2B budgets differ by0.010316%. All six trained-gradient gates pass;maximum body relative-L2 error is0.369447%,and every declared family passes. Complete state/RNG restoration,replica equality and source/work counters pass. Exact future trajectories after arbitrary restarts remain unestablished.

Actual shared/branched work is13,657 research updates and7,147,738,926 source-target presentations. All16 native phases exit successfully. MAIN uses40.299804 assigned native chip-hours;the separate whole-instance opportunity interval is42.627241 chip-hours over159.85 minutes,including CPU tails. Do not add these two cost measures. Tuning/preparation costs are retained separately.

The retained2B training paths use12.292222 chip-hours for fixed2 versus11.976412 for One,about2.64% more. This was a nominal-matrix-work comparison,not a chip-time match. The paths include their own prefix/finish phases and exclude separate scoring/audits;shared prefixes must not be double-added to actual campaign costs.

The registered independent-reference bins remain secondary. The easiest deciles slightly favor One;harder bins favor fixed2. They are target-dependent diagnostics,not routing signals or proof of reasoning. Recombining original positions reproduces pooled NLL.

Fresh training-seed confirmation and stronger eligible cheap/lean controls remain outstanding. The older mixing-only model trained varying depths;the fixed-two mixing-only control is now being tuned separately. Both original selected rates were at the upper grid edge. No TEST,reserve groups or terminal alpha are used;the broad research goal is unachieved.

[Exact complete result](report.json) · [Costs](execution-costs.json) · [Registered secondary bins](reference-loss-strata.json) · [Earlier first-pair observation](first-primary-pair.json) · [Measured prefill cost](../fixed_two_prefill_20261025/REPORT.md) · [Next cheap control](../fixed_two_mixing_control_20261026/tune/REPORT.md).
