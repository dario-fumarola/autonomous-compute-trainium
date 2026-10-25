# Completed small-core development pilot

All nine native training segments and all nine complete512-window numerical
gates finished. Independent FP32 CPU scoring covers all declared endpoints.
The primary result does not support useful recurrence at this budget.

| Arm | Cooled equal-data DEV NLL | Cooled matched-work DEV NLL |
| --- | ---: | ---: |
| Recipe ordinary | 2.615775 | 2.454545 |
| True one-pass recurrent family | 2.617206 | **2.448818** |
| Sampled recurrence,declared depth-two exit | 2.616335 | 2.450674 |

Equal-data endpoints each see1,004,887,590 authentic source targets. Matched
work endpoints see3.6463B/3.5966B/3.3339B targets respectively and each completes
its own registered cooldown. One-pass and recurrence have exactly141,963,840
parameters;recipe ordinary has140,781,888. The one-pass/recurrent nominal
matrix budgets differ by0.00406%. This is one paired training seed1231 on
consumed DEV,not a fresh confirmation or a training-seed significance test.

At matched work,recurrence loses0.001856 nats to its parameter-exact trained
one-pass control. The paired source-group95% interval for candidate gain is
[−0.004313,+0.000240],conditional on these fitted models. Its0.003871 gain
against recipe ordinary is below the declared0.01 useful-effect threshold.
At equal data it improves only0.000871 over the one-pass control and loses
0.000561 to recipe ordinary. No endpoint justifies scaling this recipe.

Within the recurrent checkpoint,the supported one-to-two-pass improvement is
0.017189 at matched work and0.014194 at equal data. That does not override the
trained-control comparison. Depth three is secondary;the primary exit remains
depth two. The full pooled primary readouts retain their originally registered
document-hash-derived initial noise. A separately registered prefix-only-noise
sensitivity audit is reported separately;it cannot alter the primary result.

The149 authentic-code targets do not pass the joint context-use criterion.
An exploratory separately trained one-pass readout also fails to establish a
candidate benefit:matched-work gain0.025630 with repository-bootstrap interval
[−0.067164,+0.116443];equal-data gain−0.049196 with interval
[−0.140675,+0.046866]. These intervals measure source uncertainty,not seed
robustness. The control was added after the candidate readout and is explicitly
exploratory. No target,band or inference exit was searched for a win.

The previous execution failure and recovery/adoption remain in the records.
All original weights,moments,data cursors and endpoint schedules were preserved;
failed/stopped work is charged. Fresh confirmation data and alpha are untouched.
See report.json,execution-costs.json and the earlier pilot/tuning cost ledgers.

Decision:close this small-core recipe at the declared budget. The next proposed
mechanism study changes the predictive objective in a controlled2×2 design.
It is not admitted until the separately registered attention and actual-update
qualification passes. No numerical finding explains away these model scores.
