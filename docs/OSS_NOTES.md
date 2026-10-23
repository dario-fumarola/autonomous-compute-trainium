# PyTorch and Trainium observations

The post-science [fused-optimizer qualification](../results/fast_path_20261023/REPORT.md)
passes CPU agreement on all full-size parameter shapes, offset gradient views,
changing LRs, exact recovery and concurrent half-node training. A separate
direct-ATen fixture reproduces ignored overflow/unscale metadata in the pinned
backend. Its normal unscaled case agrees with CPU. High-level GradScaler is
untested; neither the completed science nor the qualified unscaled candidate
uses those optional inputs. [Reproducer and contribution note](OSS_FUSED_ADAMW_METADATA_20261023.md).

Record issues only after reproducing them against a pinned version. Separate documented limits, configuration errors and backend defects. Keep a small local reproducer, exact environment, expected/actual behavior and affected training result for any contribution candidate. Do not submit upstream issues or PRs autonomously.

## Verified setup constraint

The [native quickstart](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/get-started/quickstart-torch-neuron.html) limits standalone wheels to Ubuntu 26.04. This host is Ubuntu 24.04 with glibc 2.39; the inspected CPython 3.12 native wheel requires GLIBC_2.43. That is a documented packaging constraint, not a newly discovered correctness bug. Use an isolated Ubuntu 26.04 container and preserve the host's working XLA environment.

The [native release notes](https://github.com/aws-neuron/torch-neuronx/blob/main/release-notes/2.33.0.md) specify static compiled shapes. Use fixed windows and masks; eager execution is for correctness and fallback discovery.

Upstream source: aws-neuron/torch-neuronx at 50b14171affaa365f1df6aa015232427336f6a4d. TorchTitan fork: aws-neuron/torchtitan-neuron at 3f447e28a4cf2f8ba939024bfdd427ee8084ee8c. The upstream [AI policy](https://github.com/aws-neuron/torch-neuronx/blob/main/AI_POLICY.md) requires human involvement in contributions. Local research code and reproducible evidence come first.

## Reproduced: eager compiler failure does not reach the caller

Environment: PyTorch 2.14.0+cpu, torch-neuronx 2.14.3.0.282+dc70ffa, neuronx-cc 2.28.7631.0+8f7b0cf5, native runtime 2.35.30.0, host driver 2.30.2.0, Ubuntu 26.04 container, Trainium2 LNC=2.

Run `scripts/repro_native_compile_failure.py` in a fresh process. It deliberately passes an invalid compiler flag. The compiler reports a failure, but the eager matrix operation returns a tensor and CPU transfer completes without an exception. The expected 3×3 product differs by up to 260.0. The script exits 1 after detecting `BUG_NO_EXCEPTION`. This confirms the earlier observation with a missing compiler shared library. Healthy compilation produces the correct result. This is a contribution candidate concerning error propagation, not a scientific model result. Preserve the logs; do not trust a logged compiler PASS/FAIL without numerical and update checks.

## DDP device indices and a validated workaround

With one visible physical core per process, `torch.neuron.set_device(0)` accepts a local index. DDP with `device_ids=[0]` requests a side stream with index 0, which native stream creation rejects on a process whose physical core is nonzero. Passing the physical index to `set_device` instead fails its local range check. Two failed 64-rank attempts are retained in costs.

Move all inputs explicitly to each process's sole device and use `DistributedDataParallel(..., device_ids=None)`. This avoids the DDP input-copy side stream. The two-rank and subsequent 64-rank checks pass: globally averaged gradients agree with a CPU global-batch reference, and full parameter/optimizer digests agree across replicas. This suggests a device-index consistency issue worth an upstream reproducer. Do not patch installed packages for the workaround.

## Compatibility observations

- Direct compiled `F.cross_entropy` produced a Dynamo data-dependent-output graph break. An explicit `logsumexp - gather` full-vocabulary CE works; CPU loss and gradient tests check the equivalence. This is a workaround pending a smaller operator reproducer.
- The pinned TorchTitan fork imports Triton from its GPU expert-parallel module while importing a dense Llama model. On a CPU-wheel environment the listed requirements omit Triton. Installing `triton==3.8.0` and `torch-remat==0.2.0` permits the import. A lazy import for optional GPU paths may be a small contribution candidate.
- The current host driver disables newer asynchronous DMA features. The measured throughput includes the runtime's supported threaded fallback. No host driver upgrade was performed.
- Single-core compiled training records no fallback operations. DDP records `aten::_to_copy`; the inventory is published without claiming zero distributed fallbacks. Model/gradient agreement and full replica checks remain required.

## Host memory during qualification

Two 64-rank pause attempts exceeded the initial 256 GiB container limit during compilation. A 416 GiB limit allowed all training updates, but compiling an additional inference graph in the same processes exceeded that limit. These are resource failures, not failed language-learning results. Every closure remains in the ledger. The sampled host reserve stayed above 551 GiB during the 416 GiB attempt; the root filesystem stayed above 35 GiB free.

The completed training benchmark uses static slices for prediction states, a shared compile cache, and `gc.collect()` plus `malloc_trim(0)` after warmup. It completes the full replica-state checks. Inference runs in a separate fresh process and passes CPU agreement with unchanged weights. Because several changes occurred together, these observations do not isolate the cause of high memory use. A smaller memory reproducer is needed before proposing an upstream fix.

One completed pause benchmark has a wrapper exit code of 127: its hardware report passed, then editing the running shell file caused a trailing command to be read incorrectly. The launcher now executes an immutable copy. The subsequent ordinary and inference runs exit 0. Preserve the original 127 receipt; do not rewrite it as a clean run.

## Packed BF16 training follow-up

The new trainer uses standard `torch.autocast("neuron", dtype=torch.bfloat16)`
with FP32 parameters and Adam moments, `DDP.no_sync` across accumulated
microbatches, and a token-correct global loss. A 64-rank tiny-model check agrees
with the CPU global accumulated gradient to relative L2 0.00110. At 150M,
ordinary BF16 loss/full-gradient CPU checks pass and 100 warm optimizer updates
measure about 526k real targets/sec, including H2D, packing and finite-gradient
clipping. Several variables changed together; this does not isolate an AMP gain.

A fresh-process 64-rank late restore still exceeded 416 GiB during its first
pause compile. Ordinary-to-pause graph coexistence was therefore not the only
source of pressure. The next attempt maps the checkpoint read-only across ranks
and returns freed CPU malloc arenas before compilation. It also verifies the
complete restored model/optimizer digest after device transfer. Keep this as a
resource-management observation until controlled profiling identifies an
upstream allocation defect. The host supervisor now watches compiler errors
and container OOM events as well as host/disk reserves; an error invalidates a
phase even when Python returned normally. No installed upstream code is patched.

The mapped/trimmed retry passed. The final context-1024, batch-2, accumulation-2
BF16 measurements over 100 warm updates are 855,336 real targets/sec ordinary
and 353,879 pause. All full replica states agree. The supervisor uses measured
phase limits instead of reserving a 416 GiB compile allowance for single-core
DEV. This fixes one conservative preflight rejection; it does not weaken the
512 GiB host reserve.

## Strided integer inputs and attention shape limits

The batch-2 trainer exposed a compiled-input prologue CPU round trip for an
int64 token view with shape `(2, 1024)` and stride `(1025, 1)`. The physical
int64-emulation warning prints `(2,1024,2):Int:S(2050,2,1)`. Its Python lowering
raises `Attempting to use FunctionalTensor on its own` in
`python_ops/contiguous_empty_hlo.py`, then falls back. Contiguous CPU tensors
before H2D avoid that path. Both complete 104-update model/optimizer results
remain bit-identical, and the extra contiguous fallback disappears. The local
`repro_noncontiguous_input.py` exercises this input contract without a model or
research data; its aggregate receipts distinguish numeric success from fallback.
Both standalone runs pass numerical agreement: the strided case records
`aten::contiguous[prologue_cpu_roundtrip]`, while the contiguous control records
an empty fallback list. This reproduces the avoidable transfer independently
of the larger model's additional FunctionalTensor warning.

The pinned backend's `_can_use_nki_flash_attention_backward` in
`neuron_dynamo_backend/decompositions.py` requires `S % min(512, S) == 0`.
The ordinary length 1024 satisfies it; pause-expanded 1152 does not. The forward
gate can accept 1152. This is source evidence for a shape-related performance
hypothesis, not a profile proving which kernels or costs dominate this run.
Record the measured 2.42× runtime penalty as well as the 1.132× dense matmul
estimate. A future operator profile or arbitrary-length attention improvement
could be useful upstream work. No unsupported peak-utilization claim is made.

The bounded right-tail-padding check now supplies a practical workaround:
expand pauses to 1152 as before, then append masked causal inputs to 1536.
All scored positions precede that tail. A CPU test compares prediction outputs
and full gradients to the unpadded model; a native 64-rank check passes CPU
loss/full-gradient agreement and complete replica-state equality. Over 100 warm
updates, throughput rises from 353,879 to 695,948 real targets/sec. The source
gate and speedup motivate an arbitrary-length backward-attention contribution;
they still do not constitute a kernel profile isolating the entire speedup.
The scientific budget counts all 1536 positions, including unused padding.
