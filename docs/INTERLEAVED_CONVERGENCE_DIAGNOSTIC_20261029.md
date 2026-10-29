# Learned-state convergence diagnostic — registered before tuning scores

The active learning-rate comparison remains unchanged. This separate CPU-only
diagnostic tests a prerequisite for interpreting its fixed-two-round graph as
an approximation to sequential latent-token generation. It does not select a
learning rate, train any weights, read DEV/TEST/reserved groups, or establish
reasoning or accelerator inference efficiency.

[PonderLM-2 Appendix E](https://arxiv.org/html/2509.23184v4#A5) reports dependence
on the trained numerical depth under fixed K=2, improved by random K in {2,3}.
That is a relevant risk, not a result for this implementation. Causality ensures
eventual finite-step agreement; it does not establish rapid convergence at our
learned weights. Jacobi rounds solve a fixed one-latent-per-token history and
are not an inference thought-budget curve.

Before observing any tuning quality score, freeze six 512-position TRAIN views:
the first two completely single-domain windows per web/math/code at or after
index10,000,128 in the accepted source index. Search at most20,000 windows.
These views are outside all fitted tuning prefixes and the numerical audit
windows. Preserve private token hashes and publish only indices and aggregates.
Apply exactly the same views to the seed9901 initialization and all three
audited latent learning-rate endpoints, without selecting the best endpoint.

For each view, report real-target NLL at K={0,1,2,3,4,8,16,32}; successive state
RMSE; K2-to-K8/K16/K32 forward KL; and exact sequential NLL, KL and state RMSE.
The cached CPU sequential implementation must first match original uncached
recomputation in forward, full gradients and causality fixtures. Also compare
both on the first32 source positions at each learned checkpoint, requiring
maximum per-target FP64-CE discrepancy no more than0.0002. Recompute cached
keys and values in chronological order, with a distinct entry for every source
and generated latent. Source and latent share the source rotary position.
No answer is used in generating a latent or choosing a numerical depth.

The provisional eligibility guard for further fixed-K2 sequential-mechanism
work is, in EACH domain: absolute K2/exact-sequential mean NLL difference <=0.01,
mean KL(p_K2 || p_seq) <=0.01, and mean KL(p_K16 || p_K32) <=0.0001.
These are declared engineering tolerances, not literature constants or evidence
of a useful effect. Report every arm regardless of its guard. Failure blocks
an automatic fixed-K2 extension or cheap-sequential-inference claim. A mixed-
round follow-up needs its own numerical qualification and frozen comparison;
neither this diagnostic nor a favorable tuning score admits it automatically.

Use at most16 CPU threads,16GiB RSS,20minutes per endpoint and90minutes total,
while retaining the512GiB host and35GiB disk reserves. No accelerator allocation,
optimizer update, new source acquisition, or full vocabulary array publication.
Keep failures, timing and all aggregate rows. CPU timing is diagnostic cost,
not a serving benchmark. Source/plan hashes are fixed before execution; learned
checkpoint hashes and their independent audit receipts are bound when available.
