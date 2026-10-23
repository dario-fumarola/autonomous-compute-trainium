# Frozen TEST confirmation after twelve terminal models are fixed

Scope amended before any TEST access. The original nine-model scope is
[archived](archive/FROZEN_TEST_CONFIRMATION_NINE_MODELS_20261023.md).
Keep TEST sealed until all original and follow-up fits, independent numerical
audits and DEV evaluator calibrations pass. Include all three ordinary/shared
1B pairs, all three untied 1B endpoints, the ordinary/shared seed-23 3B pair,
and the ordinary seed-23 annealed estimated-FLOP-matched endpoint. No missing
model may be silently dropped, and no projected weights or untrained inference
depth is eligible. The follow-up seeds and compute endpoint are declared in
[the separate protocol](CONFIRMATION_FOLLOWUP_20261023.md).

Calibrate the generic sharded evaluator on DEV against scalar scoring for
ordinary, shared and untied templates: weighted-NLL agreement within 0.0001,
CPU-forward agreement, complete target coverage, unchanged weights and exact
replica identity. Independent full-panel FP32 audits use tolerance 0.0002.
Freeze full checkpoint SHA256s, trained depths, source SHA, evaluator, analysis
and imported analysis helpers before opening TEST. Only the trained path is
allowed. Every declared endpoint is included regardless of its DEV score.

The original ordinary/shared three-seed comparison remains primary. Use the
fixed 60/20/20 real-token NLL, paired-seed Student-t 95% interval with df=2,
and useful-gain threshold 0.01 nats. A lower bound above the threshold supports
a useful gain, an upper bound below it rules it out at this exposure; otherwise
the comparison is inconclusive. Untied seed23 motivated replication: report
combined three-seed results as descriptive and fresh seeds47/71 separately
with df=1. Report every paired gain and all domains. Longer-exposure and
resource-matched controls are one seed; source-group bootstrap intervals
condition on those fitted weights, not training-seed variance. Estimated
matmul matching is not a total-cost match. Untied is intentionally larger.

After TEST release there are no additional architecture, LR, endpoint,
inference-depth or model choices on this panel. Complete and publish this
fixed comparison. Further variants require a separate held-out panel. Pure
software performance qualification cannot select models using TEST scores.
