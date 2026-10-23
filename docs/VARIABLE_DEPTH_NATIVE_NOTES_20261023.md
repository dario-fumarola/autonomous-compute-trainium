# Native-backend notes from mixed-depth qualification

These are executed software checks, separate from the unfinished scientific
comparison. [Measured qualification](../results/variable_depth_20261023/qualification.json).

A static graph per DDP worker can implement a variable-depth objective without
retaining three large compiled graphs on every worker. Ten/twelve/ten workers
compile one/two/three passes, respectively, and a stateless within-update
permutation routes existing natural windows. Global real-target normalization
is essential: DDP averages worker gradients, so each local loss sum is scaled
by world size divided by the complete update's real-target count. Native tiny
global gradients agree with CPU at relative L2 0.00145. All three full-size
path probes pass and every replica's model/optimizer state is identical.

The rank-zero path is only depth one. Checking rank zero alone would miss
failures in the longer compiled paths. The qualifier checks representative
ranks at each depth and inspects every replica's fallback list. The known
`aten::_to_copy` transfer appears during intentional host reporting and state
hashing; it is allowed explicitly. Other fallbacks fail the gate. An initial
attempt rejected that expected copy and is retained as a failed attempt.

Equal mean depth does not imply equal runtime. The mixture sustains 435,788
real targets/sec per eight-chip job, versus 577,029 for fixed depth two.
Longer workers delay the collective. These rates each cover 100 warm updates;
they exclude cold setup/final hashes and do not include periodic scientific
checkpoint cadence. They are measured budgets, not evidence of model quality
or peak hardware utilization. No dynamic scheduler is added to this study.

The new common optimizer uses beta1=0.9, beta2=0.95 and matrix decay0.1,
with zero embedding/norm decay. The pinned TorchTitan optimizer factory also
uses beta2=0.95 and weight decay0.1 in its Adam-family defaults;
our group exclusions are explicit experiment choices.
[Pinned factory](https://github.com/aws-neuron/torchtitan-neuron/blob/3f447e28a4cf2f8ba939024bfdd427ee8084ee8c/torchtitan/components/optimizer.py).
A numerical fixture covers every distinct 150M parameter shape, offset gradient
views, two decay groups, changing LRs and exact optimizer continuation; maximum
CPU parameter error is 2.98e-8. The native fused implementation passes LR and
step as runtime tensors. Actual tuning phases retain near-qualification rates
through changing warmup/decay values, rather than recompiling each model step.
[Pinned optimizer](https://github.com/aws-neuron/torch-neuronx/blob/50b14171affaa365f1df6aa015232427336f6a4d/torch_neuronx/python_ops/torch_mlir/ops/optimizer.py).

Exact serialized-state recovery is different from an identical uninterrupted
trajectory. Recovered weights and moments are bit-identical, and two fresh
processes continuing the same checkpoint produce identical final states.
Nevertheless, interrupted versus uninterrupted tiny training differs by a
maximum 3.31e-5 in parameters after eight more updates, global relative L2
1.11e-6. Step counters, source cursor and exposure agree. The new recipe's
numerical continuation gate was specified before measuring it; this result is
not described as bit-exact uninterrupted replay.

The drift's cause has not been isolated. A useful separate engineering probe
would compare the first post-restart gradients before Adam, reducer bucket
layout, compiled numerical ordering and device RNG state. The model has no
explicit dropout and routing is stateless; this does not identify the cause.
No vendor defect is inferred from these observations. The separately documented
[optional AMP-metadata gap](OSS_FUSED_ADAMW_METADATA_20261023.md) is not used by
these unscaled runs. No upstream issue or PR has been submitted.
