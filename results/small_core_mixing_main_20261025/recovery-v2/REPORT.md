# Completed cheap-mixing control

All four registered endpoints and six independent readouts are complete. The
models share original random initialization,parameter inventory,source order,
optimizer,precision and noise/depth draws within seed. Cheap mixing repeats the
adapter but executes the transformer core once. No teacher,generated corpus,
RL,new objective or changed endpoint enters this comparison.

| Model | Real training targets | Seed5231 DEV NLL | Seed7283 DEV NLL | Mean |
| --- | ---: | ---: | ---: | ---: |
| Original one-pass NTP | 1.005B | 2.614669 | 2.615013 | 2.614841 |
| Cheap mixing | 1.005B | 2.615881 | 2.611430 | 2.613655 |
| Recurrent NTP,depth2 | 1.005B | **2.592239** | **2.589951** | **2.591095** |
| Cheap mixing,matched nominal work | 1.228B | 2.583691 | 2.578843 | 2.581267 |
| Original one-pass,matched nominal work | 1.238B | 2.581290 | 2.581382 | 2.581336 |

Recurrence beats the cheap equal-data control by0.023642 and0.021480 nats.
Their conditional95% source-group intervals are[0.021686,0.025620] and
[0.019642,0.023346]. The extra core contributes value beyond this mixing
control. Cheap mixing's mean benefit over original one-pass is only0.001186,
and its direction differs between seeds. It does not recover the useful gain.

Recurrence nevertheless loses to the work-matched cheap control by0.008548
and0.011107 nats. Its corresponding conditional intervals remain entirely
negative. It also loses to the previously completed longer one-pass controls.
There is no training-efficiency win. The cheap work budget is0.023765% below
recurrence's;nominal matrix work is not equal chip-time or free inference.
Every branch completes its own cooldown from an untouched stable prefix.

The same cheap model changes only0.0000103/0.0000128 nats between supported
mixing counts1 and2. Posthoc all-dimension parameter aggregates find mean
linear state-retention factors0.00344/0.00375 for cheap mixing versus
0.14232/0.14365 for recurrence. This is compatible with near-insensitivity to
another mixing application;it does not establish a causal optimization
explanation or contraction of the nonlinear core.
[Exploratory parameter aggregates](exploratory-injection-aggregates.json).

All six complete512-window gradient audits pass;maximum body relative-L2
error is0.2997%. Complete model/optimizer/RNG branch recovery and all32-replica
state equality pass. The152 frozen source hashes are unchanged. The retained
training path performs6,614 global optimizer updates across the two seeds,
sharing only stable prefixes between the short/long conceptual endpoints.
[Completion audit](completion-audit.json) · [Exact outcomes](report.json) ·
[Measured costs](execution-costs.json).

These are two paired development seeds on consumed DEV,not independent
confirmation. Source-group intervals condition on the fitted models and do not
measure training-seed variance. No TEST,reserve,terminal alpha or terminal
success is used. This closes the cheap-mixing diagnostic at its registered
budget;it does not automatically admit another architecture or exposure.
