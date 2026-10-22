# Natural-text learning curve v1

Frozen before tuning or inspecting curve-cohort predictions. This narrows the
next result to the owner's question: does computation available from random
initialization improve prediction of unseen natural text?

Compare ordinary and early-pause 150,436,608-parameter models at approximately
100M, 300M and 1B real targets. Use three paired initialization seeds: 23, 47,
71. Use one frozen data order (data seed 23) across arms and seeds. Paired seed
uncertainty therefore measures initialization variance conditional on this
corpus/order. Start fresh; never reuse qualification or tuning model weights.
The late arm remains a follow-up about introduction timing, with its exposure
confound still acknowledged. No larger model or synthetic corpus is involved.

Use pinned natural web/math/code sources in a 60/20/20 real-token mixture. Keep
whole documents, group splits by web host/code repository, globally deduplicate
normalized exact text, and screen five-word near duplicates with a disclosed
32-value minhash/LSH plus bottom-256 Jaccard estimate. This approximate lexical
screen does not guarantee absence of all semantic duplication. Publish source,
tokenizer, filtering, count and array hashes without source contents.

Divide DEV source groups deterministically into disjoint **tune** and **curve**
cohorts. Only tune is evaluated during LR selection. After selecting and freezing
LRs, evaluate curve at all main milestones. Keep TEST sealed throughout this
first result. Check split/cohort source groups and duplicate screens before
launch. Neither curve nor TEST determines the LR.

Use the qualified context 1024, local batch 2, accumulation 2, 64-rank topology,
BF16 autocast with FP32 masters/moments, global clipping 1 and native unfused
AdamW. Before tuning, benchmark an optional causal right-tail padding of pause
inputs to length 1536: it permits a native attention fast-path without changing
earlier real predictions. Adopt it only after CPU output/full-gradient checks,
replica equality and at least 100 warm updates show a throughput benefit. Count
all padded positions in the dense FLOP estimate. Ordinary remains length 1024.
This changes hardware execution, not the 16 pauses per 128 input positions or
the source targets. Freeze the decision before any tune or curve evaluation.

For each arm, independently test peak LRs **3e-4, 1e-3, 3e-3** using seed 11,
the same first 384 global updates, and the same tune cohort. This grid shifts
one geometric step above the earlier proposal to test higher rates at the
larger batch, retaining the already-qualified 3e-4 candidate. Predeclare it now;
do not expand it based on curve scores. Warm up for 16 updates; cosine-decay
from update 346 through 384 to 10% peak. Select the lowest fixed 60/20/20 tune
NLL among finite, passing candidates. Preserve failed trials and costs. No
curve predictions are inspected until this selection is frozen.

Main fits use 3,840 global updates, with curve evaluations at 384, 1,152 and
3,840. Convert all boundaries to actual real targets from the frozen packing
prefix counts before launch. Warm up for 40 updates; hold stable through 3,456;
cosine-decay to 10% peak at 3,840. Both arms follow this same schedule policy at
matched actual real targets. The intermediate points are both stable-trunk
points, and both final endpoints complete decay. Keep complete continuation
states. A later longer fit may fork an undecayed earlier checkpoint with a new,
explicit continuation protocol rather than silently reheating the endpoint.

The primary statistic is ordinary minus early weighted curve NLL at the final
endpoint, with the predeclared paired-three-seed 95% interval. Lower bound above
0.01 nats supports a useful gain here; upper bound below 0.01 rules out that
size of gain here; otherwise inconclusive. Intermediate curves and domain
breakdowns are descriptive, with no selection of a winning milestone. Report
source-group uncertainty separately. Roughly 1B targets is about 6.7 tokens per
parameter: any negative applies only to this early exposure, not convergence.

Report actual training/evaluation/tuning wall time and all dense matmul FLOPs,
including head, pauses and optional tail padding. Token-matched curves answer
the first question. No FLOP-efficiency conclusion is made until separate fully
annealed equal-FLOP endpoints exist. No claim of self-generated thoughts, pure
introduction timing, autonomous placement, or BDI follows from this experiment.
