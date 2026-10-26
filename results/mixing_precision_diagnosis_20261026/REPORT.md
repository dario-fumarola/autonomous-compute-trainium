# Mature mixing precision diagnosis: localized repair passes

All variants use the same unchanged failed checkpoint, original 512 authentic
TRAIN windows and exact supplied latent states. No optimizer updates occurred.
The original independent gradient-family threshold remains 2%.

| Native arithmetic | Overall gradient error | Injection/readout error | Gate |
| --- | ---: | ---: | --- |
| Original FP16 mixing | 0.695354% | 4.407005% | Fail |
| Adapter alone moved to FP32 | 0.426550% | 0.594890% | Pass |
| Full body FP32 | 0.064210% | 0.052329% | Pass |

The original adapter A_log and dt_bias named gradient errors are 14.0996% and
11.9076%; adapter FP32 lowers them to 1.3655% and 1.1697%. The C readout error
is already 0.5127% originally. This localizes the observed mismatch to precision
in the mixing path. It does not establish an SDK defect, identify a single
operator as its cause, or explain the scientific quality ranking.

One independent pinned-upstream full-FP32 gradient covers the complete global
objective; the full-FP32 comparison reuses it only after all state, objective,
source, routing, supplied-noise and count identities agree. NTP scalar differences
are below 0.0000036 nats for both new paths. All model/optimizer readbacks,
unchanged-weight and complete gradient-replica checks pass. The conditional
readout-plus-adapter third capture is unnecessary and did not run.

The two native closures cost **0.444500 assigned chip-hours**. Separate whole-
instance opportunity cost, including the independent CPU tail, is **2.272074**;
these are not additive. CPU comparison phases take 385.27 and 15.01 seconds.
The earlier CPU wiring screen takes 42.796054 seconds; its initial cross-version
random-state preparation failure is preserved separately.

This is a zero-update numerical diagnosis. It does not repair the FP16 model's
training history, qualify a new training/optimizer recipe, complete the stopped
control study, or admit another scientific fit. The original study stays stopped.
No TEST, fresh reserve, terminal alpha or terminal success.

[Complete result](report.json) · [Costs](execution-costs.json) · [Plan](plan.json) ·
[Protocol](../../docs/MIXING_PRECISION_DIAGNOSIS_20261026.md).
