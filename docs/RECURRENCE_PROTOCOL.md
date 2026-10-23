# Fixed recurrence on natural text — 2026-10-23

**Superseded after the bounded tuning/learning gate.** All six sandwich trials
remain near 7.3 NLL and nearly context-independent; no 1B fit launched. The
active [pre-norm protocol](PRELN_RECURRENT_PROTOCOL.md) retains the healthy
ordinary model exactly as its single-pass control. The original version of
this proposal is preserved in [the screen archive](RECURRENCE_SANDWICH_SCREEN_20261023.md).

This is a new mechanism study. The completed pause experiment stays frozen.
The current campaign is bounded to eight hours on the existing instance.
No synthetic research examples, external teacher, discrete thoughts, RL,
learned halting, or social auxiliary are introduced.

## Question and falsifiable outcome

Does applying a shared causal transformer core twice, from initialization,
improve unseen natural-text prediction over applying it once? The primary
comparison uses equal original source targets, identical order, initial weights
and unique parameter count within each of seeds 23, 47 and 71. A positive gain
means single-pass NLL minus two-pass NLL. At the fully annealed 1,004,887,590
target endpoint, a useful gain requires the lower 95% paired-seed Student-t
bound to exceed 0.01 nats/token; an upper bound below 0.01 rules out that gain
at this exposure. Otherwise the result is inconclusive. Three seeds do not
establish convergence, broad reasoning, or discovery of a thinking policy.

## Architecture

Use the qualified TorchTitan causal attention/MLP primitives, 768 width,
12 query / 4 KV heads, 2048 FFN width, 32k tied vocabulary, context 1024.
Partition the 20 unique blocks into 2 prelude, 16 core and 2 coda blocks.
Use RMSNorm before each attention/MLP and after each residual addition
(sandwich normalization), with a final RMSNorm and vocabulary projection.
Compute e = Prelude(embedding), s0 = 0, s_i = Core(s_(i-1) + e), then
head(Coda(s_r)). Backpropagate through every application. There is no detach.

These are published additive-injection, zero-state and SandwichBlock options
from [the recurrent-depth authors' implementation](https://github.com/seal-rg/recurrent-pretraining/tree/1ea7220ec7eb42d13e89db0663df254d0bcdc28e).
This is a deliberately fixed, deterministic screen, not a replication of
[Huginn's stochastic-depth training at 800B tokens](https://arxiv.org/abs/2502.05171).
Both primary arms use exactly 150,467,328 unique parameters. The single-pass
arm is a conventional 20-block causal composition with sandwich norms;
the two-pass arm has 36 block applications. Added post-residual norms and
input injection are shared design choices, not unacknowledged changes to the
older Llama-style ordinary baseline. That older baseline is contextual only.

An optional secondary untied-depth control clones each core block into an
independent second-pass block, matching the initial two-pass function and
effective depth/FLOPs but using more parameters. It is a sharing control,
not a parameter-matched comparison. Run it only if measured remaining time
allows its own equal-budget LR selection and a full seed-23 endpoint.

## Data, tuning and evaluation

Reuse the immutable 1.05B natural web/math/code corpus and its existing
group-disjoint TRAIN, tuning DEV, curve DEV and sealed TEST partition.
The same frozen packed stream is read without wrapping. All short documents
and tails remain included. Artificial boundary tokens carry no source loss.
Corpus/source identities and hashes are recorded in the launch specification.
The previous pause results used curve DEV; therefore new curve results are
development evidence, and the first sealed TEST readout is separately gated.

Select each architecture's LR independently on tuning DEV using the same
three candidates (3e-4, 1e-3, 3e-3), seed 11, 384 updates / about 100M targets,
16 warmup updates and final 38-update cosine decay. Freeze all choices before
the first main curve evaluation. Do not tune on curve DEV or TEST.
Main runs use 40 warmup updates followed by a stable trunk. Save descriptive
curves at updates 384 and 1152, and a stable checkpoint at 3456. Independently
branch at 3456 to a 384-update cosine decay ending at update 3840, floor 0.1.
Intermediates are not annealed and are not endpoint comparisons.

Primary metric: 60/20/20 weighted domain NLL, nats per real target. Also report
all three domain NLLs, individual paired-seed gains, and source-group bootstrap
intervals conditional on the fitted checkpoints. These are distinct uncertainty
sources. Paired seeds share one corpus/order, not independent data foundations.

Before any sealed TEST access, lock all primary weights, LR choices, token and
compute endpoint definitions and analysis code hashes. Score both fixed primary
arms on the same TEST documents once; keep model selection on DEV. Do not make
new variants or select checkpoints based on TEST. A failed small-model result
does not authorize scaling. No original 1.7B official TEST is released here.

## Compute control and schedule

Report equal-token and equal-FLOP results separately. Count full vocabulary
projection even though tied, all recurrent applications, attention, backward
and any recomputation. Publish the analytical dense-matmul assumptions and
measured warm throughput, compilation, checkpoint and evaluation wall time.
The FLOP control compares single-pass update 3840 with a shorter two-pass
endpoint q = floor(3840 * F_single / F_recurrent), where F is the estimated
training FLOPs of one full global update. Declare q before main results.
Fork a final 10% cosine decay from the stable recurrent trunk at
q - ceil(0.1*q). Restore that untouched stable trunk for the token-matched
continuation. Both compute-matched endpoints complete their own decay.
The small rounding difference is reported. Additional branches are campaign
cost, not counted as though free. This comparison intentionally gives the
ordinary model more unique source targets at equal estimated FLOPs.

## Systems gate and budget

Reuse BF16 matmuls with FP32 masters/Adam moments, global real-target-weighted
DDP accumulation, clipping 1, atomic full-state checkpoints and immutable
launch snapshots. Use fresh workers per shape or evaluation. Fail closed on
compiler errors, OOMs, numerical disagreement, replica disagreement or reserves.
Keep host available memory above 512 GiB and every output filesystem above
35 GiB free. A separate tmpfs holds new artifacts; it was created at 96 GiB
and expanded to 160 GiB before tuning to retain all full curve/branch states.
This is filesystem capacity, not reserved physical RAM. The shared container
memory ceiling includes its new checkpoint pages. Old results remain.

Before scientific training: actual-wrapper CPU causal/gradient/sharing and
full optimizer-continuation checks; native tiny global-batch reference; native
150M all-64-rank numerical agreement, full-state recovery, replica equality;
at least 100 warm updates per primary arm. Compare full-node and half-node
training throughput with constant global batch where practical. Select the
topology by measured aggregate targets/sec and include all attempts in costs.
Freeze a measured budget after qualification, with an eight-hour overall cap.
Stop spending on a mechanism when the declared endpoint answers this screen;
do not expand an unsuccessful recipe into a broad search.

## Exploratory diagnostics and checkpoint transfer

Before reading new per-position pause scores, fix bins by source target index
j (one-based within each original 1024-target window): pre-first-block j<128;
then offsets j mod 128 of 0, 1–7, 8–31, 32–63 and 64–127. Compare existing
ordinary/pause endpoint models on identical curve DEV targets. Reconstruct
the original aggregate exactly within numerical reduction tolerance. Treat
this as a locality diagnosis, not a replacement success criterion.

Inspect the retained matched 1.7B raw/trace states for a bounded natural-data
transfer readout. A common adaptation recipe and its natural dataset/splits
must be frozen before scoring those checkpoints. Original synthetic traces
are historical state, never a new corpus. Execute this secondary study only
without displacing the qualified primary campaign or weakening its gates.
