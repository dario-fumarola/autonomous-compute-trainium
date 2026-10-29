# Isolating persistent generated memory

The current generated-state model improves broad development prediction at
equal data. A trained pause control tests whether generated states help beyond
constant inputs. Neither comparison isolates whether future predictions need
access to earlier generated states. The bracket-type diagnostic remains
inconclusive; it is not a reason to claim a recovered reasoning procedure.

The proposed control retains the same six-block body, input/output tables,
initialization, source scale, rotary indices and one generated state per source
position. It first computes ordinary causal source representations. A generated
state then passes through the shared body, attending to its authentic source
prefix and itself. Source positions cannot attend to any generated state, and
generated positions cannot attend to other generated states. The resulting
state predicts the next authentic token. No gradients are detached.

This removes both direct and indirect propagation of latent content across
source positions. It changes the model's learned memory access, not merely its
readout. It must therefore be trained as a separate control before any causal
interpretation; applying its mask to an existing persistent model would be an
out-of-distribution intervention.

The CPU reference recomputes each prefix plus its own generated state. An
equivalent parallel masked implementation matches predictions and all parameter
gradients. A second implementation reuses ordinary source keys and values,
with gradients through that reuse. Tests check optimizer moments, future-token
invariance, zero influence from other latent inputs, and actual dispatched
forward/backward matrix work. Software test arrays are not research data.

Unlike persistent latent feedback, this graph has no sequence of unknown
earlier generated states to solve, so it needs no Jacobi approximation. Both
methods still perform one generated-state step per source token at inference.
This control might lose predictive quality, or retain it at lower training
cost. Neither outcome is assumed.

## Disposable hardware qualification only

Wait until all six active replication fits pass their mature audits and their
Native workers close. Their quality results are not a qualification gate;
their CPU evaluation may run independently. Do not change those frozen fits,
sources, numerical gates or evaluation code.

Qualify the actual 51,747,328-parameter, six-layer, width-512 shape on all
16 chips/64 ranks, two 512-position TRAIN windows per rank, FP32. Use the
already accepted disposable qualification source, seed 40271 and LR 0.0003.
Check the cold full gradient against the independent CPU reference before any
update, then measure 100 warm updates. Check mature gradients on the complete
128-window global batch. Preserve the existing 2% per-family gradient and
0.003 loss tolerances. Verify all replicas and exact model/optimizer/RNG replay
in a separate process. Every qualification state is excluded from science.

The explicit entry point installs only the registered model factory and exact
matrix-work accounting in unchanged qualification engines. Its Native path
reuses source KV; its independent CPU path recomputes expanded masked history.
No scientific training entry or automatic promotion is provided.

Cap the queue at 45 minutes, each Native phase at 20 minutes and CPU audit at
10 minutes. Stop if the existing 0.8-second warm-update screen fails. Retain
512 GiB host memory, 192 GiB worker cap, 35 GiB disk and 10,000 inode reserves,
plus a four-GiB artifact allowance and both projected source snapshots. Record
all startup, compilation, update, replay, audit and failed work.

No held-out panel, fresh reserve, predictive model comparison, router or longer
fit is admitted. A later trained comparison must independently tune the control
and separate equal-data, training-work and inference-cost conclusions.

## Relation to published work

PonderLM-2 motivates the persistent generated-state reference. Trading Depth for
Time separately studies explicit cross-token feedback in a different backbone.
Our proposed removal of latent-memory attention is not that paper's feedback
ablation and does not inherit its findings. It is a specific control for this
repository's computation graph.

[PonderLM-2](https://arxiv.org/html/2509.23184v4) ·
[Trading Depth for Time](https://arxiv.org/html/2609.21605v2).
