# Does matching memory during learning matter?

The frozen corruption diagnostic preserves89–94% of the broad prediction
gain under unrelated same-domain memory. This new bounded learning control
tests training association rather than adding depth or searching readouts.
It starts a separate scientific history;all prior negative gates remain.

Train one new one-step branch from the original seed7285 and zero output,
with exactly the same655360 parameters,frozen C0/head,8192 target-source views,
2048 updates,batch4,32 uniformly sampled natural labels,query RNG7286,
AdamW LR3e-4/decay0.1 and clipping1.0. No reset,LR search,cooldown or
score-based extension. Do not initialize from an already trained branch.

At every training query keep the original current-token representation and
its own original memory slot. Replace earlier memory with a same-domain
TRAIN view from a different source group. Use a deterministic bijection
within each domain:sort by source group/index and rotate by the largest
group's view count. Verify all pairs are group-disjoint and every donor view
appears exactly once. Refuse if a group exceeds half a domain. Source metadata
alone determines the pairing. The original current state still contains
backbone context;this does not remove all information about the past.

Targets and target-source order are unchanged. Each arm supervises8192 views
and262144 targets. This control additionally accesses8192 donor memories from
the same8192-view feature bank;there is no additional unique corpus. Donor
accesses and the extra projections count in work. This is equal updates and
parameters,not an equal-total-work claim or a deployable donor-bank policy.
No target-aware choice,teacher,generated target or new inference router.

After a mature audit,evaluate both the saved authentic-memory-trained branch
and the new independent-memory-trained branch with authentic memory and TWO
fixed donor maps from TRAIN. Freeze those maps before training using seed7297,
same-domain donors from two distinct groups. Both models see the same maps.
Keep all448 reused TUNE views/116 groups and the60/20/20 aggregate. Raw C0 and
the saved local nonlinear control are reference scores. The prior authentic
endpoint must reproduce within1e-5;no held-out outcome selects the endpoint.
The320 unscored reserved groups are excluded from training,donors and evaluation.

Primary contrast is new-independent-training NLL minus original-authentic-
training NLL under AUTHENTIC evaluation memory. For each donor map also report
both models' matching-memory utility and their interaction:does authentic
training improve the benefit of authentic evaluation memory? Report the full
two-by-three table,domains and conditional recipient-source intervals. Donor
maps are fixed;these intervals do not include training-seed or donor-pool
uncertainty and are not confirmation. A lack of difference is not proof of
equivalence. A one-seed positive is only a development lead.

For a useful content-learning lead require the primary advantage>=0.005nats
with a positive lower95% bound,plus positive matching-memory utility for the
original model and positive interaction under BOTH donor maps. This is an
attribution screen,not a replacement for the original0.01 absolute/0.005
iteration gates. No outcome admits a larger backbone,router or confirmation
automatically. If the control matches or improves prediction,the current
attention gain cannot be presented as evidence that source-matched memory
learning caused it.

Use an algebraically equivalent efficient donor implementation;compare every
parameter gradient to the separately computed per-query replacement reference.
Before fitting,qualify at the existing mature one-step weights on one fixed
TRAIN view per domain,FP32 versus FP64 at the unchanged2% gate. Those weights
are disposable and never scientific initialization. Repeat the actual-path
audit at the new endpoint. Verify full model/AdamW/counter/RNG restoration and
one disposable next-update replay. Preserve scientific checkpoint hashes.
Count the vocabulary projection and its backward pass in measured matrix work.

Reuse all frozen features;no new Native feature pass. Bound registration to
30minutes,training to20minutes and audit/evaluation to the remainder. One
16-thread CPU fit. Measure the first16 new updates before admitting the rest;
retain and charge them. Keep512GiB host availability,35GiB disk,10000 inodes
and48GiB process RSS. Preserve all failed work/full states. No fresh reserve,
TEST,old confirmation pool,or new dependency evaluation is scored here.
