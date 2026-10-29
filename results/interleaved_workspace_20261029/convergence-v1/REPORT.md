# Learned-state consistency: two endpoints fail; highest LR passes

The initialization passes all three domain guards. The first audited latent
endpoint (LR0.0003,128,025,288 targets) fails them. Higher-round states converge
to the independently implemented exact sequential history, but the trained
two-round readout is meaningfully different.

| Domain | Exact sequential NLL minus K2 NLL | KL(p_K2 || p_sequential) |
|---|---:|---:|
| Web |+0.037709|0.055073|
| Mathematics |+0.073237|0.116940|
| Code |+0.111294|0.103735|

These are two frozen TRAIN windows per domain, outside fitting and numerical-
audit ranges. They are not a held-out predictive comparison or an estimate of
general reasoning capability. Both NLL discrepancy and KL exceed the0.01
eligibility tolerances. K16/K32 KL is below1.1e-9 in every domain: eventual
convergence does not rescue a low-round approximation.

![Numerical-depth consistency at the first learning-rate endpoint](latent-lr0-consistency.png)

The short-prefix cached/recomputed reference checks pass. No weights change
during this40.48-second CPU diagnostic. The full active tuning comparison is
unchanged and no predictive scores have yet been released. The other two latent
endpoints remain pending. No automatic fixed-K2 extension or cheap sequential
inference claim is admitted. A separately qualified mixed-round comparison is
the next specific hypothesis; it is not assumed to repair this mismatch.

[Frozen protocol](../../../docs/INTERLEAVED_CONVERGENCE_DIAGNOSTIC_20261029.md)
and [first-endpoint aggregates](latent-lr0.json).

## Second audited endpoint, LR0.001

The identical diagnostic takes40.21 CPU seconds and again fails the guard in
all domains. Exact-sequential minus K2 NLL is+0.002844 on web,+0.040639 on math,
and+0.091553 on code. Forward KL is0.058263,0.077923 and0.085006 respectively.
The smaller aggregate web loss shift does not imply close prediction agreement.
K16/K32 KL remains below1.2e-7. The third endpoint is pending; no learning-rate
selection or quality promotion follows from these TRAIN-only diagnostics.
[Second-endpoint aggregates](latent-lr1.json).

## Completed third endpoint, LR0.003

The third endpoint passes the unchanged guard in every domain. Absolute mean
K2/sequential loss shifts are0.001404 on web,0.000939 on mathematics and0.000142
on code. Forward KL is0.000642,0.001066 and0.000601 respectively; K16/K32 KL is
below7.3e-13. This establishes that the observed consistency failure depends
on training dynamics; it is not inevitable for the fixed-round architecture.
It does not establish useful prediction, generated-content value or efficiency.
All three endpoints and the initialization are now evaluated under the same
frozen rule, without selecting a learning rate from these TRAIN examples.

| Learning rate | Maximum domain absolute NLL shift | Maximum domain forward KL | All-domain guard |
|---|---:|---:|---|
|0.0003|0.111294|0.116940|Fail|
|0.001|0.091553|0.085006|Fail|
|0.003|0.001404|0.001066|Pass|

The active tuning protocol still selects by its predeclared broad DEV loss,
which is pending along with the pause controls. No higher-LR quality win is
assumed from this consistency result. A mixed-round follow-up remains separately
gated; no scientific fit is automatically admitted by this diagnostic.
[Third-endpoint aggregates](latent-lr2.json) and [complete receipt](report.json).
