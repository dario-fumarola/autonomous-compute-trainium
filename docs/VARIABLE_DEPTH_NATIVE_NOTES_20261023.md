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

A trained100M-checkpoint continuation probe later reaches1.11% native/CPU-BF16
gradient relative L2. The frozen ongoing trainer guard is8%; startup hardware
qualification used1% and passed before fits. No threshold or algorithm is
changed. A CPU-only calibration on that exact natural TRAIN batch finds
**1.017% CPU-BF16/FP32 gradient difference** in the independent implementation.
Its FP32 loss4.144055 differs from native4.144054 by about1.4e-6. This shows
precision sensitivity at that state; it does not directly measure native-to-
FP32 gradient error, because the native gradient vector was not retained.
[Calibration](../results/variable_depth_20261023/cpu-precision-calibration.json).

## Prospective topology qualification

`depth_topology.resize_depth_signature` and its CPU tests establish a useful
invariant:32 ranks with local batch4 and16 ranks with local batch8 assign
every original window to the same recurrent depth, with the same two
microbatches, routing and accumulated objective. The numerical fixture also
reproduces the global masked gradient. A matching global batch alone is
insufficient:64 ranks/batch2 and8 ranks/batch16 change the mixture and are
rejected. No active scientific signature or tensor state is resized.

After the current science and read-only annex finish, a separate bounded
performance qualification can compare the16-rank layout on the same eight
physical chips per job. It uses fewer logical cores and may be slower; no
speedup is assumed. Require full-size CPU agreement at each active depth,
global-gradient normalization, replica equality, full optimizer recovery and
100 warm updates before considering future use. Preserve the running32-rank
campaign and count every qualification attempt separately.

The queued `scripts.qualify_depth_topology` controller implements a separate
35-minute, at-most-eight-attempt benchmark. It waits for successful primary
completion, terminal read-only annex completion and no live training/scoring
workers. It does not take an idle half from an unfinished scientific fit.
Tiny32/16-rank mixed-depth global-gradient checks precede fresh paired150M
ordinary/mixed benchmarks in both layouts, then full16-rank optimizer restores.
Each full benchmark measures100 warm updates; every active depth must meet
the same0.01 loss/relative-gradient qualification limits, with finite values
and identical replicas. Source-window exposure, estimated NN work and initial
weights must match across layouts. Cold setup and failed work count separately.

Promote the candidate only if actual joint pair throughput improves at least
10% and neither arm loses more than5%. This is a bounded engineering decision,
not a scientific learning result. Native core shape changes can alter floating
point trajectories; no cross-layout bit-identical trajectory is claimed. The
existing native cache symlinks already point to private instance NVMe; no
cache relocation or environment replacement is part of this benchmark.

The live scientific controller's `--resume` has narrower operational scope
than model/optimizer recovery. It recomputes the whole original budget before
reusing stages, cannot reuse a failed occupied launch identity, and reconstructs
its frozen module inventory by glob. Newly added unused modules therefore
also prevent a direct restart against the earlier inventory. The current
controller is live. If recovery becomes necessary, preserve the old receipts
and tensor states and make an explicit orchestration-only recovery amendment;
do not silently restart completed fits or change frozen models/analyses.
