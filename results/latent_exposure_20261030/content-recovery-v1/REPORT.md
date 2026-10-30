# Generated-state content and persistence matter at the saved endpoint

The complete four-condition CPU diagnostic finds strong dependence on the
model's generated-state pathway. Removing access to past latent slots raises
broad NLL by **0.636840 nats** and arithmetic answer NLL by **0.323091**. These
are effects of intervening on one trained model, with all downstream states
recomputed. They do not establish a reusable reasoning algorithm, the quality
of a separately trained memory-free model, or training-compute efficiency.

All 384 broad windows and all 259 frozen arithmetic cases are retained. No
example selection or new model fit occurs. Authentic exact execution reproduces
the previously published scores within **1.33e-15**. The longer ordinary control
continues under its original recipe, scoring rule and deadline.

| Saved-weight condition | Broad NLL | Answer NLL | First-answer NLL | Exact answer prefixes |
|---|---:|---:|---:|---:|
| Authentic generated states | **2.801700** | **2.255895** | **2.156507** | 5/259 |
| Current-only generated state | 3.438540 | 2.578986 | 2.219186 | 3/259 |
| Previous position's generated state | 6.413301 | 3.265870 | 3.566065 | 0/259 |
| Constant latent input | 9.476707 | 11.218651 | 11.413017 | 0/259 |

Current-only attention blocks every earlier latent key for both source and latent
queries. Source states are recomputed under that restriction, so latent information
cannot survive through old source caches. Current-token extra computation remains.
Lagged input uses the previous position's generated state from the intervened
history; the first position is unchanged. Constant input uses the scaled mean
embedding. All conditions retain the source/latent slots, rotary positions,
body calls and dense attention dimensions. Masking does not claim a sparse saving.

## Paired removal penalties

Positive values below mean intervention raises loss relative to authentic use.
Intervals are 95% conditional source-bootstrap intervals on one trained seed
and adaptively reused DEV data, without multiplicity-adjusted confirmation.

| Intervention | Broad penalty (interval) | Answer penalty (interval) |
|---|---:|---:|
| Current-only | +0.636840 [0.624087, 0.650573] | +0.323091 [0.267140, 0.376771] |
| Lagged input | +3.611600 [3.553417, 3.665658] | +1.009975 [0.904521, 1.116702] |
| Constant input | +6.675007 [6.600477, 6.750204] | +8.962756 [8.845882, 9.080799] |

Current-only first-answer loss increases by **0.062679**, interval
**[0.006768, 0.115324]**. Its answer-loss penalties grow across the two/three/
four-to-six-operation bins: **0.205981 / 0.355528 / 0.407764**. However, the
hardest bin's first-answer loss improves under current-only attention by
0.024607; this exception is retained. Accurate complete answers remain scarce.
Every broad domain deteriorates under each intervention.

## What follows

The generated states are used, including their cross-token persistence. Deleting
that persistence at inference does not retain the saved model's prediction lead.
This rules out presenting simple path deletion as a free optimization of this
checkpoint. It does not rule out an independently trained cheaper graph.

Large ablation penalties are not equally large gains over ordinary models. The
actual equal-data advantage over the trained ordinary model remains **0.030495
broad nats**. The separately trained pause model scores **2.901458**, whereas
injecting constant inputs into the latent-trained model scores **9.476707**.
That difference illustrates how much an unfamiliar input pathway can hurt.
Constant replacement also changes state scale, and lagging changes alignment;
neither isolates logical content while preserving its full distribution.

The original answer-NLL advantage remains predominantly in later teacher-forced
answer tokens. Five exact answer prefixes do not establish reliable arithmetic.
A trained memory control and independent replication would be stronger evidence;
this diagnostic admits neither automatically. The primary matched-work ordinary
comparison remains pending and cannot be overruled by these interventions.

## Checks, execution and costs

Twenty-four focused software tests pass. Every condition passes three mature
cached/full-prefix and batch/individual checks before full scoring; maximum
log-probability discrepancy is **7.82e-14**. Independent standard-library
reaggregation verifies every one of the **2,572** saved rows, their paired
sources/targets/bins, absolute means and answer counts. No private texts, token
arrays, case identifiers or individual scores are published.

The successful recovery takes **1,861.467 wall seconds** and **31.622 CPU-hours**
across the four workers, with peak worker RSS **3.232 GB**. Zero optimizer updates
and zero assigned accelerator chips. It overlaps the already assigned Native
ordinary fit; reservation opportunity cost must not be double-added. Concurrent
host work can slow that fit, so its resulting wall time is not an isolated
hardware-time comparison.

The first low-thread attempt stopped after **319.997 seconds**, before any
complete scientific score file, because throughput projected beyond its cap.
A low-thread numerical benchmark also hit its **240-second** cap. Both are
retained. Completed throughput checks selected execution settings only, without
reading labels or ranking model quality. Recovery changes batch/thread allocation
and dispatch bookkeeping, retaining the original **06:22:53 UTC** deadline and
all scientific fields. It completes before that deadline. Both fresh reserves
remain sealed and terminal alpha spent is zero.

[Full aggregate report](report.json) ·
[Independent reaggregation](independent-reaggregation.json) ·
[Original stopped attempt](../content-diagnostic-v1/report.json) ·
[Execution-only recovery](../../../docs/LATENT_CONTENT_BATCH_RECOVERY_20261030.md) ·
[Original diagnostic](../../../docs/LATENT_CONTENT_DIAGNOSTIC_20261030.md) ·
[Equal-data result](../learning-v1/REPORT.md).
