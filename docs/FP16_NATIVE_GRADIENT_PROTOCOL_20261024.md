# Bounded native FP16 gradient qualification

Identity: `fp16-native-gradient-v1`. Numerical qualification only, no optimizer
updates, model training, held-out scoring or terminal-test allocation.

The CPU screen passed0.7251% gradient error on eight actual TRAIN windows with
FP16 internal matmuls, FP32 tied vocabulary projection/CE and fixed loss scale8192.
Test the same precision policy on the complete256-window global objectives of
the preserved ordinary update9216 and fixed update7680 checkpoints. The ordinary
checkpoint's directory name contains10368; its **actual stored counter is9216**.
Do not mistake the filename for its contents. Preserve all source hashes.

Use32 ranks per eight-chip half, batch1/accumulation8, the same stateless routing
and real-token normalization as the existing full CPU references. Scale backward
by8192, then unscale FP32 master gradients after DDP averaging. No GradScaler or
fused-optimizer AMP metadata is used. Primitive SDPA is selected before torch
import. All masters remain FP32. Model weights/cursors and source files stay
unchanged; every native replica must have identical final unscaled gradients.

Compare the complete native global gradient and NLL against the preserved full
CPU FP32 references, with unchanged gates: gradient relativeL2≤0.02, NLL
difference≤0.003, finite gradients, correct real-target count and no unexpected
fallbacks. Pin both checkpoint and reference-file hashes before capture.

Start only after both previous scientific final native readouts close with
exit0. Give each capture its **own isolated container** cloned from the exact
existing SDK root filesystem, with the existing /work and driver-library mounts,
host networking and160GiB memory cap. The aggregate cap remains320GiB. The old
shared container is stopped after its native work closes; installed host drivers,
XLA and SDK files are preserved.

The isolated supervisor checks container ownership before launch/cancellation.
Compiler, numeric, timeout and per-container OOM failures stop only that worker
container. Shared host512GiB or filesystem35GiB reserve emergencies can stop all
managed worker containers. A real CPU-worker test has verified that canceling one
container preserves the other container and its existing worker. Historical
frozen launchers stay unchanged; this new execution uses only isolated launchers.

Cap the paired native captures and reference comparison at30 minutes. Preserve
failed cases and costs. Passing this gate does not qualify optimizer updates,
scaling/overflow handling during learning, complete-state recovery, throughput or
scientific benefit. Those remain separately bounded prerequisites for adoption.
All completed FP32 scientific comparisons retain their original interpretation.
