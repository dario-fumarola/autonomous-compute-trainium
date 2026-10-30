# Separate the immediate readout from later memory access

This is an exploratory follow-up to the negative sparse comparison and its
position decomposition. It does not repair any earlier gate. Freeze this full
two-by-two analysis before scoring these interventions; no fitting or favorable
example selection is allowed.

Use both selected audited workspace checkpoints, generated and constant, and
all 384 original DEV sources. Independently vary (1) whether the four boundary
predictions use the source representation or the following slot representation,
and (2) whether later queries can attend earlier slot keys. Preserve every
source/slot position and still compute every slot. Recompute all downstream
states after masking. No authentic cached state survives a changed attention
mask. Report all four conditions for both models, every domain and the five
previously fixed position bins.

These are off-distribution interventions. In particular, the source-boundary
plus blocked-memory condition is the same weights' ordinary predictor, not an
independently trained baseline or a supported zero-workspace deployment mode.
An improvement from removing a path diagnoses harm in that trained checkpoint;
it is not a new efficiency or reasoning result and may not survive retraining.
The original broad comparison against independently trained ordinary remains
negative regardless of any intervention.

Before full scoring, for every model/domain/worker verify four identities at
1e-8 maximum log-probability error: authentic full-prefix versus cached; changing
only the boundary readout leaves all other positions unchanged; blocking earlier
slots leaves the first 128 predictions unchanged; both switches equal the
same weights' independently executed ordinary path. Require 24 complete check
records (96 scalar checks), and reconstruct both authentic broad/domain scores
within 1e-8 afterward.

For each altered condition report authentic NLL minus altered NLL: positive
means the intervention improves prediction. Report the loss-scale two-factor
interaction as well. Source-group bootstrap intervals use 2,000 draws, seed
40515, and remain descriptive, unadjusted and conditional on repeatedly used
DEV. No slice is an advancement criterion.

Four CPU workers, eight threads each, six-GiB RSS cap per worker, 20-minute
duration and absolute bound; existing host/disk floors apply. No Native owner
may be active at execution. No retries, new training, memory gates, routers,
unsupported extra slots, fresh reserves or terminal alpha. Bind all source,
selected checkpoint and completed comparison hashes. Publish aggregates only.
