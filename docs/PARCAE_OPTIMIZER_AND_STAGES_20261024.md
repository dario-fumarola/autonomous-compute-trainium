# Parcae execution preparation and cost accounting

Separate preparation, with no change to depth-connection-v1. The active
scientific comparison continues its own FP32 recipe and frozen endpoints.

## Completed CPU checks

The release-style optimizer candidate passes five prescribed gradient steps in
each of three modes: plain AdamW/eager Muon, fused AdamW/eager Muon, and fused
AdamW/AOT-eager Muon. The maximum parameter difference is6.88e-6 relative to a
reference step's size; Muon moment differences are zero. Complete nested state
matches on load and after an exact next-step replay. Gate grouping explicitly
uses the released-code `release_muon` policy. A separate `paper_adamw` grouping
is available but has no scientific or native qualification.

This CPU qualification takes4.465s including imports. It uses numerical gradient
fixtures, not a research corpus. The BF16 polar iteration is part of the specified
optimizer; its masters and moments remain FP32. There is no implicit loss scaling
or optional fused AMP metadata. Callers must check finite global gradients and
unscale before clipping/stepping. Missing gradients fail before either optimizer
family changes any state.

[Exact optimizer checks](../results/parcae_optimizer_cpu_20261024/attempt2-state-and-aot/report.json) ·
[CPU execution receipt](../results/parcae_optimizer_cpu_20261024/attempt2-state-and-aot/execution-costs.json).

Static encode/refine/decode stages pass CPU AOT-eager with `fullgraph=True` and
`dynamic=False`. Ordinary1-pass and recurrent1/1,2/2,4/2 and3/1 forward/backward
depths match the previously checked monolithic model exactly in logits and full
gradients. Checkpoint names remain unchanged; model transfer still uses the
standard PyTorch module path. This takes12.796s including imports. The first
attempt failed to save its report because a parameter variable shadowed the
argument object. Its failure stays preserved; attempt2 supplies the evidence.

[Stage checks](../results/parcae_stages_cpu_20261024/attempt2/report.json) ·
[Preserved first failure](../results/parcae_stages_cpu_20261024/attempt1/failure.json).

The SDK documents compilation of models/functions and separate AOT forward and
backward graphs. Composing our stages still requires actual native validation;
CPU success is insufficient. [Official TorchNeuron architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.33.0/torch/about-pytorch-neuron.html).

## Independent global reference and queued native gate

Pinned upstream CPU code has accumulated the complete256-window natural-TRAIN
FP32 objective,261,664 real targets, for both ordinary and recurrent4-forward/
2-backward execution. Each also matches our port on a complete1024-token window.
All parameter gradients are saved privately. Initial hidden states, weights,
source packing and reference files have pinned hashes. The weights include an
explicit numerical perturbation; they cannot initialize a scientific fit.
These chance-level TRAIN fixture losses are numerical readouts, not model quality.

The preparation takes448.089s from launch to terminal report, with zero chips
and zero optimizer updates. It overlaps existing native training. The existing
32k tokenizer is preserved, an explicit adaptation from the published32,768
vocabulary. The actual parameter counts are140,781,888 and141,963,840.

[Independent CPU references](../results/parcae_native_global_20261024/cpu-report.json) ·
[CPU receipt](../results/parcae_native_global_20261024/cpu-execution-costs.json).

The registered native queue waits until depth-connection-v1 completes and all
managed native workers release their chips. It then uses both eight-chip halves
for32-rank captures, with a30-minute execution limit and no optimizer updates.
It cannot start after01:00 UTC October25. Local/full gradient and NLL gates,
source/weight hashes, replica equality, fallback inventory and ownership-scoped
cancellation are mandatory. Twelve source files are frozen by its execution
plan; adding this candidate changes none of the47 scientific frozen files.

[Native protocol](PARCAE_NATIVE_GRADIENT_PROTOCOL_20261024.md) ·
[Registered execution plan](../results/parcae_native_global_20261024/native-execution-plan.json).

The first waiting-only scheduler was explicitly canceled before any native
attempt. Review found that a ready prerequisite could bypass its latest-start
check. Execution revision2 checks the cutoff before readiness and at both launch
decisions; a regression test covers late readiness. The old plan/source and
zero-chip cancellation receipt remain preserved. Only the scheduler changed;
all CPU vectors and model/capture sources are identical. Runtime status now
distinguishes waiting, running and terminal execution.
[Unstarted revision1 receipt](../results/parcae_native_global_20261024/unlaunched-r1/cancellation.json).

No scientific fit, fresh held-out scoring or terminal attempt is launched here.
Native optimizer updates, full-state recovery, sampled-depth DDP and100 warm
updates with measured throughput remain prerequisites for a new research budget.

## Extra matrix work is material

The new optimizer contains matrix multiplications that ordinary AdamW does not.
Count them, including redundant execution on every DDP replica. The formula
matches an independent PyTorch executed-operation-counter fixture.

At context1024 and global256 windows:

| Graph | DDP ranks | Muon matmuls / model forward-backward matmuls |
| --- | ---: | ---: |
| Ordinary | 32 | 21.69% |
| Ordinary | 64 | 43.38% |
| Recurrent4/2 | 32 | 15.43% |
| Recurrent4/2 | 64 | 30.87% |

The64-rank geometry holds the global batch fixed by halving accumulation. Neural
work stays identical; redundant optimizer work doubles. These are matrix counts,
not latency percentages: the optimizer's polar matmuls are BF16 while the proposed
body qualification is FP32. Embedding gathers, pointwise AdamW, communication,
idle time and compilation are still separate costs.

Future budgets should include model and optimizer matmuls and measure elapsed
time. Profile larger accumulation/context before settling the training batch.
The existing published experiments retain their declared cost definitions.
[Exact structural work](../results/parcae_work_accounting_20261024/report.json).
