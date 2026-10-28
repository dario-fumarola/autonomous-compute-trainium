# Does more source diversity improve residual-workspace transfer?

The completed512-update residual screen failed its original gates. Training
corrections were larger than held-out corrections. This new bounded development
experiment tests source diversity against additional optimization,without
changing the mechanism,learning rate or original success thresholds. It is
registered after that negative result;it is not a retroactive extension of it.

For each linear/one-cell/two-cell mode,restore the exact512-update model,
AdamW state,counter and target-sampling RNG. Fork into two schedules:

| Schedule | Total updates | Unique source views | Traversals |
| --- | --- | --- | --- |
| Fresh | 2048 | First8192 TRAIN views | One |
| Repeated | 2048 | First2048 TRAIN views | Four |

All six arms add1536 updates with four views and32 uniformly sampled target
positions per update. Total262144 target exposures include the original65536.
Source order remains fixed;no reshuffle or source selection by model outcomes.
The two schedules use identical target-position RNG at each update. Repeated
views can receive new sampled target positions. View count is not a claim of
that many independent documents:report source groups and domain counts.

Keep frozen C0/head,branch widths/initialization,LR3e-4,AdamW decay0.1 and
clipping1.0 unchanged. Keep the512-step prefix costs. No LR search,cooldown,
early stopping or checkpoint selection. The causal memory and query semantics
are those of FROZEN_RESIDUAL_SCREEN_20261028.md. The branch does not generate
discrete notes and its intermediate exits are not trained inference endpoints.

Reuse the existing frozen first2048 TRAIN and448 TUNE feature caches. Verify
their exact content hashes and source-array receipts before admission. Extract
only the next6144 TRAIN views using the same160-position Native graph,all16
chips and the three independent HF reference checks. All8192 TRAIN entries
must be source-group disjoint from the fixed448-view TUNE panel. The latter is
reused development evidence,not fresh confirmation;upstream foundation data
exposure is not independently known. No sealed DEV/confirmation data is used.

Primary mechanism gates stay unchanged:the fresh two-step model must improve
full32 held-out NLL over frozen C0 by at least0.01nats,with positive conditional
source-group lower95% bound;it must also beat both fresh linear and one-step
controls by at least0.005nats with positive lower bounds. A separate source
diversity screen requires fresh two-step to beat repeated two-step by at least
0.005nats with positive lower bound. Report every corresponding contrast for
all modes,including failures. No multiplicity-adjusted confirmation claim is
made from these exploratory gates. A fresh-data benefit without an iterative
benefit supports data diversity,not learned iterative reasoning.

Frozen raw NLL must reproduce2.282111260797275 within1e-5 and agree across
all six arms. Use the same60/20/20 weighting and source-group bootstrap as the
previous screen. No intermediate held-out scoring. Every final endpoint must
pass all named FP32-vs-FP64 branch gradients at the unchanged2% threshold and
exact model/optimizer/RNG restoration. A disposable TRAIN diagnostic repeats
the original512-view checks;it does not modify scientific checkpoints.

This is equal-update/equal-source-exposure branch learning. Fresh/repeated
conditions in each mode have the same matrix work per update. Extra feature
extraction for fresh views is measured and charged separately,so this is not
equal total training cost. Across modes,report actual vocabulary/head-backward
and branch work,parameters and CPU time. Shared prefixes and feature work are
counted once as actual campaign cost and identified on each logical path.
No claim of beating full ordinary continuation is available from this screen.

Bound admission to45minutes,Native extraction10minutes,CPU arms30minutes
each and final CPU diagnosis10minutes within the original overall deadline.
Six16-thread CPU workers fit the small branches. Measure the first16 new
updates to verify the fixed endpoint fits the budget. Retain512GiB host
MemAvailable,35GiB disk reserve,10000 free inodes and192GiB Native worker cap.
Require9GiB additional disk allowance before launch. All failures and partial
updates count;no automatic recovery budget or score-based extension. Preserve
all earlier scientific states. No router,full-model scaling,from-scratch or
confirmation campaign is admitted by this registration.
