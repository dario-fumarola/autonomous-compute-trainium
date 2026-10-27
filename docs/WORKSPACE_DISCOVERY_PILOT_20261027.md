# First bounded workspace learning comparison

This is a single-seed, constant-rate, short continuation diagnostic. It is
neither a converged efficiency comparison nor confirmation of reasoning or
learning from random initialization. No full pretraining or router follows
automatically. The stopped recurrence program stays stopped.

All four conditions start from the identical provenance-pinned stage-one
OLMo2 common C0 weights, with fresh AdamW. C0 includes the already recorded
30-update reader calibration; all calibration, failures, qualification,
sampling, scoring, reconstruction and audit costs remain in the research ledger.
Qualification-updated weights are excluded. No teacher supplies thoughts.

Use the frozen60/20/20 web/math/code new-tokenizer source views. Each begins
with128 authentic prefix tokens and32 authentic continuation tokens. Source
order is shared, with no wrap, extension or outcome-based filtering. Longer
controls get more source views. No condition reads TUNE during fitting.

The workspace arm has48 global64-view updates, G4 independently sampled
eight-token workspaces, four-token reward horizon, fixed length cost.001,
and detached leave-one-out advantages. The objective is.5 ordinary131-target
NTP +.5 mean reader loss +.1 whole-trajectory policy loss. The same frozen
EMA scorer computes with/without-workspace scores and updates only after the
actor, decay.99. Log absolute gains; positive centered advantage is insufficient.
The writer reads prefix+opening cue and never receives the future. The reader
reads workspace+unchanged prefix without reader-side prose; empty work is
exactly ordinary input. Both prefills are charged.

Controls at the same nominal matrix-work allowance, rounded UP:

| Condition | Global updates | Objective and execution |
| --- | ---: | --- |
| Ordinary |931| Ordinary NTP on all159 next-token targets;160 physical positions |
| Workspace |48| Above hybrid;192 physical positions |
| Fixed pauses |421|.5 ordinary131-target NTP +.5 four-token readout after8 fixed EOS positions following the prefix |
| Draft |63|.5 ordinary NTP +.5 mean reader loss on four OWN sampled8-token drafts +.1 supervised8-token draft writer loss |

The draft's authentic future tokens are writer TRAIN targets only. Its reader
and every evaluation always use its own samples. The pause token is an existing
trainable EOS row, not the padding row. Ordinary and pause have no generated
policy, scoring copy or unused head. Every arm gets one1e-5 LR opportunity,
FP32 body/masters, AdamW(.9,.95), epsilon1e-8, decay.1, clip1 and scale256.
There is no duration selection, LR sweep, convergence claim or inference-length
extrapolation. This constant-rate diagnostic must not be presented as the
old charter's cooled, independently tuned terminal comparison.

The ordinary condition uses uniform NTP rather than the reader's concentrated
four-token weighting. Pause and draft share that reader weighting. A favorable
workspace result would still need an ordinary control with matching target
emphasis before attributing its entire benefit to intermediate computation.

The nominal matrix ledger includes the complete dense body, full-square causal
primitive attention, every projected vocabulary row, every sampled alternative,
both scorer paths, backward and checkpoint recomputation. Count FMA=2. This
is NOT all-operation or hardware-time matching: softmax, normalization,
collectives, AdamW, EMA, data transfer and sampling are included in whole measured
time. Whole costs, source positions actually consumed, repeated target labels,
generated tokens and qualification are reported separately. The current
uncached workspace update costs about19.38 ordinary updates under this ledger;
do not assume a2–3x overhead.

Before activation require the actual-size100warm diverse-source workspace
qualification and independent full179-parameter mature audit. Qualify the new
ordinary160-position graph against an independent eager HF-body/FP64-loss full
gradient BEFORE its first update. Every condition saves complete model,
optimizer and RNG state plus a zero-update full gradient on a frozen TRAIN
source group absent from all fitted prefixes. The workspace also saves its
scorer. Exact ZIP compression changes no tensor bytes. All conditions must
pass the unchanged2% named-gradient mature audit before ANY held-out scores.
Any numerical/compiler/resource failure stops the queue; no silent retry.

Evaluate320 source-only-selected TUNE anchors, exactly60/20/20, with one
prefix-only eight-token sample per anchor and fixed per-anchor seeds. Never
best-of-G, target-aware selection or resampling. Report four-token and first-token
NLL on the deployed path, raw32-token NLL, and raw four-token NLL. Compare
the workspace with ordinary, trained pauses and the draft arm; compare all
absolute scores with C0. For workspace/draft/C0, recompute matching, different-
source-group donor and fixed-content workspaces at the same length. A privileged
true-future copy is a separately labeled reader positive control, never policy
input or a deployable thought. No cached states survive content edits.

There is no preregistered confirmation threshold in this small diagnostic.
Positive within-model utility must not result from a deteriorated raw path;
positive utility alone does not establish content-sensitive transfer. Any
advantage over ordinary remains exploratory until comparable tuning, independent
training seeds, fresh sources and inference cost are addressed. A null applies
to this recipe and budget. Preserve every outcome and decide the next experiment
from the complete table, without automatically extending the most favorable arm.
TUNE source-group intervals, if reported, are conditional on this trained
checkpoint; upstream base-pretraining contamination cannot be excluded.

All16 chips/64 ranks,192GiB worker allowance,512GiB host reserve and35GiB
filesystem floors remain mandatory. Register phase/source/data hashes, caps,
complete snapshot inode allowance and full next-endpoint storage before launch.
No private prefixes, tokens, thoughts or states enter public Git.
