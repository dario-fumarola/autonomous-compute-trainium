# Exact sequential mixed-round readout: predictive lead retained

The previously K2-selected mixed checkpoint completes exact sequential latent
generation on all384 existing broad DEV windows (196,608 natural targets).
No weights, learning-rate choice, source rules or target selections change.

| Predictor/readout | Web | Math | Code | Weighted NLL |
|---|---:|---:|---:|---:|
| Selected mixed, exact sequential |3.762500|2.928660|2.097509|**3.262734**|
| Selected mixed, finiteK2 |3.763593|2.929225|2.097728|3.263546|
| Selected mixed, finiteK3 |3.761979|2.927876|2.096782|3.262119|
| Independently tuned ordinary |3.869100|3.049119|2.229139|3.377111|
| Independently tuned trained pauses |3.895149|3.053362|2.216832|3.391128|

The exact sequential gain is **0.114377nats over ordinary**, conditional95%
source interval **[0.108939,0.119788]**, and **0.128394 over trained pauses**,
**[0.123140,0.133584]**. Every domain favors generated latent states.
The sequential andK2 means differ by only0.000812nats, with sequential slightly
better. The original fixedK2 model's0.058892-nat sequential penalty is absent
here. SupportedK3 is0.000615nats better than exact sequential; neither finite
approximation is substituted for the deployed sequential result.

This is evidence that a persistent, self-generated continuous state can learn
useful natural-text prediction computation from random initialization in this
small screen. The separately trained pause control has equal nominal inference
matrix work; it does not match the candidate's larger training work. The mixed
candidate uses5.592856times ordinary nominal training matrix work, one seed and
reused development sources. These conditional intervals do not account for
training-seed variation, LR selection or adaptive research decisions.

General transferable reasoning, useful multiple-thought budgets, learned
allocation, conventional-frontier superiority and training efficiency remain
unestablished. K2/K3 are numerical approximation rounds, not thought-token
budgets. Both fresh reserves remain unscored. The stronger tied-table ordinary
challenger is independently tuning; its result is not yet available.

All six independent cached/full-prefix FP64 checks pass, maximum4.44e-15nats
per target. Two CPU workers complete in746.87seconds, peak1.864GB per worker,
zero accelerator chips and zero optimizer updates. This is not an optimized
serving benchmark. [All comparisons](report.json) · [Frozen plan](plan.json).
