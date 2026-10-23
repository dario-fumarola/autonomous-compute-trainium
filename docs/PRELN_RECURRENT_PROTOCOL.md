# Pre-norm fixed recurrence — 2026-10-23

This supersedes the [sandwich-normalization screen](RECURRENCE_SANDWICH_SCREEN_20261023.md).
All six 100M-target sandwich LR trials completed at 7.335–7.337 NLL and their
predictions were almost context independent on natural prefixes. No 1B fits
launched. Every checkpoint and failed launch gate is retained. Hardware and
gradient correctness did not establish healthy language learning.

## One controlled question

Does a second application of a shared transformer core, available from random
initialization, improve unseen natural-text prediction? This is sequential,
differentiable computation; it supplies no external information or textual
thought labels. No synthetic corpus, teacher, RL, gates or halting are used.

Keep the qualified 150,436,608-parameter ordinary model: 768 width, 20 pre-norm
causal Llama-style blocks, 12 query/4 KV heads, 2048 FFN width, tied 32k head,
RoPE, context 1024. Partition its existing blocks into 2 prelude, 16 core and
2 coda layers. Compute e = Prelude(embedding), s0 = 0, s_i = Core(s_(i-1)+e),
then final_norm(Coda(s_r)) and the tied vocabulary projection. Use r=2, with
full gradients through both passes. No detach or activation recomputation.

The r=1 case is exactly the original ordinary model, rather than another
normalization recipe. Actual-wrapper CPU checks establish bit-identical
outputs, full gradients and initial weight hashes. The full-size seed-23
initial hash matches the preserved control. The two-pass model has the same
parameters and initial weights, with 36 rather than 20 block applications.
Additive input injection and pre-norm are supported small-scale options in the
[recurrent-depth authors' code](https://github.com/seal-rg/recurrent-pretraining/tree/1ea7220ec7eb42d13e89db0663df254d0bcdc28e).
This is not a replication of Huginn's larger stochastic-depth training recipe;
the paper explicitly distinguishes its small-scale normalization ablations
from its final large model. [Paper](https://arxiv.org/abs/2502.05171).

## Reuse the matched ordinary controls

Reuse all three completed ordinary runs (initial seeds 23/47/71) at updates
384/1152/3840. Do not discard or select a seed. They used the same tokenizer,
source stream, random initialization, BF16 matmuls, FP32 parameters/Adam
moments, clipping, 64 ranks, batch 2 and accumulation 2. The new recurrent runs
use that same topology and optimizer recipe. Every new initial weight hash
must match its paired ordinary run. Prior ordinary LR selection used the same
three candidates, 100M targets, tuning seed 11 and tuning DEV that the new
recurrent selection receives. Its fixed selected peak is 1e-3.

This is reuse of deterministic, qualified controls, disclosed as such. The
old pause models are additional contextual references only. Reusing ordinary
controls saves training cost; it does not make their historical compute free.
Keep reused costs separate from the incremental campaign and report both.

## Frozen exposure and tuning

Use the exact frozen 1.05B natural 60/20/20 web/math/code stream, including
short documents and tails, without wrapping. Match every real target, order,
packing boundary and per-target loss coefficient. Preserve source-group split
and global duplicate screening. Tuning, curve DEV and sealed TEST stay separate.

Recurrent LR candidates are 3e-4, 1e-3 and 3e-3, seed 11, 384 updates,
16-update token warmup and 38-update final cosine decay. Select minimum fixed
60/20/20 tuning DEV NLL. Preserve every trial, and freeze the selection before
any new curve evaluation. A finite loss alone does not qualify learning:
the selected 100M-target recurrent trial must have tuning NLL below 5.0.
If it fails, stop this mechanism screen before 1B fits. This learning gate is
declared before any pre-norm recurrent trial; the healthy ordinary tuning
reference is 4.0513 NLL. Do not add more LR candidates to rescue a failed gate.

Main runs warm up over the same first 40 updates as ordinary, then train at
the selected peak. Save stable curves at 384 and 1152 updates, and preserve
stable trunks at 2042 and 3456. Fork the 3456 trunk into a 384-update cosine
decay to update 3840 (1,004,887,590 real targets), floor 0.1. The ordinary
schedule has the same peak/warmup/stable/decay timing, with its independent LR.
An explicit future decay in the old control and no decay boundary in the new
trunk give the same schedule values before 3456. Never resume an annealed
endpoint for further trunk training.

## Endpoints and uncertainty

Primary: 60/20/20 held-out real-target NLL gain, ordinary minus recurrent,
at the annealed 1,004,887,590-target endpoint. Across the three paired initial
seeds, support a useful gain only when the lower 95% Student-t bound exceeds
0.01 nats/token; rule out that size of gain if the upper bound is below 0.01;
otherwise report inconclusive. Conditional source-group bootstrap intervals,
domain NLLs and individual seeds are separate readouts, not extra independent
backbone replications. One fixed corpus/order does not estimate data-order
variation, and this exposure does not establish convergence.

The existing curve DEV has been inspected in earlier research; treat these
curves as development evidence. Before sealed TEST scoring, freeze weights,
all endpoint definitions and the analysis source hashes. Score the three
ordinary/recurrent endpoint pairs on identical TEST documents without model
selection or new variants afterward. Keep that confirmation separate.

## Equal-FLOP comparison

The full-global-step dense matmul estimates are 286,044,821,913,600 FLOPs
ordinary and 483,956,914,913,280 FLOPs recurrent (1.69189×). These include
the vocabulary head, all block applications, full-square attention and backward
at twice forward; they exclude elementwise, optimizer work and hardware-counter
claims. Report measured training and inference time as well.

Compare ordinary update 3840 with recurrent update 2269: 593,774,218 real
targets, 0.0286% less estimated training FLOPs due to full-update rounding.
Fork a separate 227-update cosine decay from recurrent stable update 2042
(534,370,272 targets), floor 0.1. Both compared endpoints are fully annealed.
Resume the untouched 2042 trunk for the token-matched continuation. Count
this extra branch as campaign cost. A token-matched win is not an efficiency win.

## Bounded longer-exposure diagnostic

Before the first primary curve, reserve a secondary seed-23 ordinary/recurrent
continuation to update 11520 (about 3B real targets), conditional only on a
qualified append-only natural corpus and measured time fitting the original
11:12 UTC campaign deadline. This is a single-pair learning curve, not another
three-seed claim. Do not select this continuation based on primary gain.

Retain all original packed tokens/masks byte-for-byte as a prefix. Append only
new naturally sourced, globally deduplicated TRAIN documents. Reconstruct the
original near-duplicate index from pinned source shards; require all original
document identities to be recovered, screen appended documents against it,
and retain original DEV/TEST. Freeze source hashes and order before fitting.
Resume the ordinary seed-23 stable checkpoint at update 3456 and the recurrent
seed-23 stable trunk at 3456. Preserve all optimizer moments and counters.
An explicit schedule-extension gate may move a future decay only when the
recovered checkpoint has not started annealing, while keeping peak, warmup
and floor unchanged. Verify the full original data prefix and recovered
exposure before allowing a changed corpus fingerprint. Use stable training
through update 10368, then a fixed 1152-update decay to 11520.
Qualify actual native full-state recovery on the expanded corpus first.
If the data/time gate fails, report the continuation as unexecuted.

## Systems and resources

Use the isolated native TorchNeuron environment; preserve host drivers and XLA.
Require actual CPU/native loss/full-gradient agreement, tiny global-batch
agreement, full model/optimizer recovery and replica equality. Fail closed on
compiler errors, OOMs and resource reserves. Measure 100 warm updates of the
actual path before budgeting. Initial sandwich benchmarks found concurrent
eight-chip jobs 22% faster in aggregate; this does not automatically transfer
to this model or the latency of three jobs. Match the existing 64-rank controls
for the primary study and use all 16 chips for training.

The previously unused fourth instance-local NVMe was verified to have zero
writes, no filesystem signatures, mounts or open users before preparing new
scratch. Old filesystems and results are untouched. Runtime caches were
copied, hash-verified and redirected there, with their original copies retained.
This storage is not an off-instance backup. Maintain 35 GiB free on every output
filesystem and 512 GiB host available memory. Record all stopped prototypes,
tuning, compilation, probes, evaluation and checkpoint costs. Do not launch a
large model or broad search. An optional untied-depth sharing control requires
its own declared LR budget and complete endpoint within the same time cap.
