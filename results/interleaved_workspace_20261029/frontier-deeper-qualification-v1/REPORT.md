# Deeper ordinary numerical qualification

The pre-inventoried ten-layer,width448,seven-head,FFN1112 ordinary model has
51,654,848 parameters with independent32k input/output tables. It passes cold
CPU/Native gradients,100 warm updates, all64 replicas, a fresh-process exact
update104 replay from full model/optimizer/RNG state, and the independent
128-window mature global gradient audit. Warm updates average0.272427s.

Disposable qualification weights are excluded from science. No DEV score is
produced here. The independently tuned equal-data screen is separately frozen
and now running on all16 chips, with its paired analysis registered before
predictive outcomes. Host, worker, disk and inode guards are unchanged.
[Full qualification](report.json) · [Costs](execution-costs.json).
