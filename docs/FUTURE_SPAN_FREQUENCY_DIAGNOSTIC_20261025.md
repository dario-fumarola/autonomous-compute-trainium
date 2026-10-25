# Exploratory future-loss diagnostic

This does not alter MAIN,the NTP secondary or conditional work-control admission.
It was registered after six final numerical audits and the first seed's complete
quality outcomes. It cannot provide confirmatory evidence for the objective.

The completed first-seed audits report future presence loss around6.70 in the
zero-weight controls and3.55 in the future-supervised models. A lower auxiliary
loss does not establish useful internal computation or improved language-model
prediction. Check an inexpensive reference before interpreting that difference.

Fit one context-independent Bernoulli probability for each vocabulary label,
using presence counts from the first16,384 original TRAIN windows. Use fixed
Jeffreys smoothing `(count + 0.5)/(anchors + 1)`. Do not tune the prior,sample
size,horizon or target. Evaluate on the exact next512-window TRAIN batch already
used for all original mature gradient audits,disjoint from the fitting prefix.
No model receives this reference as input or supervision.

Counting takes28.65 CPU seconds,zero model forwards and zero optimizer updates:

| Context-independent reference | Normalized future presence loss |
| --- | ---: |
| Fixed initial31/32000 probability | 6.696829 |
| Fitted TRAIN-prefix frequency | 4.483896 |

The frequency fit uses16,747,994 real source targets and1,001,657 valid anchors.
The probe has523,350 targets and31,277 anchors. Future labels remain t+2…t+32,
with document/window boundaries and duplicate-label presence handled identically
to the original objective. Raw counts remain private;only aggregate losses,
sizes and canonical count hashes are published.

The registered observer waits for all eight original models and numerical gates,
then reports every model's already collected independent objective loss against
these references. These are sampled-training-depth TRAIN forwards,not the
fixed-depth DEV scores. The ordinary NLL primary and auxiliary loss have
different denominators and must not be compared as a common quality metric.

A model beating this constant predictor would not by itself establish a causal
context-use or refinement mechanism:the predictor uses a limited fitting prefix,
whereas the network has trained on the full stream. Nor do differences from a
uniform prior causally partition the network's improvement into frequency and
context components. This diagnostic can characterize auxiliary-task learning;
only the unchanged controlled language-model comparisons can establish transfer.

No new TEST access,terminal allocation,data generation,teacher or scientific
training. [Registration and reference](../results/future_span_frequency_20261025/plan.json).
