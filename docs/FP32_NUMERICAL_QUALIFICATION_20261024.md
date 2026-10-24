# FP32 numerical qualification and recovery scope

This note defines the numerical basis for a **new, separately registered**
FP32 development execution. It does not resume the stopped BF16 campaign,
declare its guards passed, or relabel the failed strict future-trajectory test.
A scientific execution still needs its own frozen protocol and measured budget.

## Qualified mathematical path

Use FP32 masters, activations and matmuls with the SDK primitive SDPA route,
selected before torch import. Preserve the optimizer, clipping, source order
and source-token normalization. The independently accumulated full global CPU
reference uses the exact same256 TRAIN windows and checkpoint.

| Model/source | Native-vs-CPU global gradient relativeL2 |
| --- | ---: |
| Fixed, update7680, batch1/accumulation8 | 0.000120684 |
| Fixed, update7680, batch2/accumulation4 | 0.000121337 |
| Ordinary, update9216, batch1/accumulation8 | 0.000130255 |

All clear the unchanged2% gradient and0.003 absolute loss gates. Native
replicas agree exactly. Isolated primitive operators and a separate public
SDPA reproducer support the route selection. This does not qualify the
previous full-model BF16 gradients, which remain discrepant.

## Exact state load versus future arithmetic

Full model and optimizer tensors, RNG, data cursor, schedule and counters are
verified exactly at checkpoint load, including after transfer to the device.
Replicas must still finish each phase with identical complete model/optimizer
states. No moment, source exposure or schedule is discarded to obtain a pass.

The additional strict test comparing108 uninterrupted updates with4+104
updates **failed**. All counters, schedules and RNG match; resulting parameter
relativeL2 is0.0002965. That test stays recorded as a failure.

The subsequent registered short diagnosis finds:

- Two independent uninterrupted8-update runs are bit-identical.
- Independent fresh4-update prefixes are bit-identical.
- A4+4 restart differs from uninterrupted8 by4.12e-8 parameter relativeL2 and
  2.30e-6 first-moment relativeL2. RNG and all training metadata match.

Independent whole-DEV FP32 CPU scoring of the108-update trajectories gives
2.511792390 versus2.511779442, a difference of-0.000012948 nats. That clears
the **separately preregistered**0.0001-nat sensitivity limit, about770 times
smaller than the0.01-nat useful research effect. It does not make the strict
bit-identity test pass. These qualification states are not scientific endpoints.

One plausible source is DDP's different startup and rebuilt communication
buckets. The installed torch2.14 code initially uses an unrestricted bucket
and later calls the reducer's rebuild path. Changing floating-point grouping
can change rounded results. The [DDP source](https://github.com/pytorch/pytorch/blob/main/torch/nn/parallel/distributed.py)
and [PyTorch numerical-accuracy note](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)
describe the relevant mechanisms; neither proves that bucket rebuilding caused
this particular observation. No such causal claim is made here.

## Conditions for a new development run

Require exact complete state load, the qualified full-global FP32 mathematical
path, the existing2% local full-gradient probe before each phase, finite
gradients and optimizer states, identical replicas, and exact source counters.
Record phase boundaries and any restart. Do **not** promise bit-identical
future trajectories across process boundaries. Source-order/RNG preservation
and independent training-seed replication remain necessary controls.

Independent whole-panel FP32 scoring is required for the new endpoints. The
0.0001 qualification sensitivity limit is an engineering readout, not a
research success threshold or a substitute for fresh confirmatory seeds and
source groups. Keep the original0.01 research threshold and repeated-testing
charter unchanged. No TEST was read by this numerical diagnosis.

The decision to use these explicitly scoped requirements is prospective for
the new execution. Prior stopped protocols, failed tests and their costs stay
immutable. The purpose is to obtain learning curves on a mathematically
qualified path while reporting the measured reproducibility limit honestly.

[Global/reference reports](../results/trained_gradient_diagnosis_20261024/REPORT.md) ·
[Repeat/restart diagnosis](../results/trained_gradient_diagnosis_20261024/controls/fp32-repeat-floor-report.json) ·
[Independent NLL sensitivity](../results/trained_gradient_diagnosis_20261024/controls/fp32-recovery-nll-report.json).
