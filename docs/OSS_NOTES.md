# PyTorch and Trainium observations

The small-core scheduled-training pass found avoidable changing-scalar overhead:
native `fill_.Scalar` specializes its value,while tensor-copy inputs preserve
the same optimizer mathematics. The runtime-input policy passes independent
actual-update/global-gradient/recovery checks and100 varying-schedule updates.
The first20 scientific model/optimizer states match the old execution exactly.
[Policy and evidence](PARCAE_RUNTIME_SCALARS_20261024.md) ·
[Qualified rates](../results/parcae_runtime_scalars_20261024/recovery-v2/REPORT.md).
This is a performance observation,not a claim of incorrect optimizer outputs.

A separate pinned reference API accepts but does not apply an optional attention
mask on its default mixer path. A tiny CPU fixture needs no private data;
ordinary scientific causal scoring is unaffected.
[Local contribution note](OSS_REFERENCE_MASK_ARGUMENT_20261024.md).
Neither finding has been submitted upstream.

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

## October24: topology qualification stopped before model execution

The separate fixed-topology qualification's16-rank mask selected two LNC2
cores per chip over eight chips. It failed in process-group initialization
with `failed to init a collective algorithm: no_hier no_mesh`; its32-rank
contiguous control passed. This is not a model, optimizer or loss failure.
No speedup was established, and the current scientific continuation retains
the established contiguous32-rank halves. The failed receipts remain in
[the qualification report](../results/fixed_topology_qualification_20261024/report.json).

The pinned2.33 documentation has a potentially useful clarification target:
the [environment-variable catalog](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.33.0/neuron-runtime/api/nrt-env-var-reference.html)
shows a noncontiguous core-list example, while the
[runtime developer guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.33.0/neuron-runtime/guides/nrt-developer-guide.html)
requires consecutive indices in its allocation discussion. Neither page, as
inspected, establishes support for the particular distributed replica group
we attempted. In our one-core-per-process setup, each process's individual
allocation is contiguous; the collective group is not. Thus the guide alone
does not prove that our cross-process group must work or must fail.

Our inference is an unsupported/undocumented placement constraint until a
minimal init-plus-allreduce reproducer distinguishes allocation from collective
group support. An improvement could be an explicit supported-placement table
and an early actionable error identifying the offending group, rather than
a generic algorithm-initialization failure. This is a local documentation and
error-message opportunity, not an established runtime correctness defect or
an upstream submission. No extra hardware test was launched while all sixteen
chips train the mature-exposure comparison. A contiguous16-rank/four-chip
placement remains unqualified; no throughput or memory-fit claim is made.

## October24: reproduced FP32 SDPA precision gap

The mature-model backward guard exposed a precision gap worth upstream review.
The installed native backend accepts FP32 Q/K/V in its default NKI SDPA path.
The underlying `nkilib/core/attention/attention_cte.py` documentation says it
uses FP32 for softmax and BF16 for other operations, including when input dtype
is FP32. This is a documented kernel design; whether automatic selection under
PyTorch's `highest` FP32 policy should use it is the upstream question.

The public [checkpoint-free reproducer](../scripts/repro_sdpa_fp32_precision.py)
uses eight independent one-core processes, causal GQA with12 query/4 KV heads,
sequence1024, head dimension64, seed87319 and `highest` matmul precision.
HLO caches are disabled. Each pair uses identical numerical Q/K/V and upstream
derivatives, with NKI enabled or disabled before torch import. CPU FP32 and
FP64 references agree closely. These arrays are software fixtures, never
research examples.

| Q/K input standard deviation | NKI gradient error vs CPU FP64 | Primitive gradient error vs CPU FP64 |
| --- | ---: | ---: |
| 1 | 0.001162 | 0.00000207 |
| 2 | 0.009754 | 0.00000171 |
| 4 | 0.046955 | 0.00000334 |
| 8 | 0.197181 | 0.00001255 |

These are full Q/K/V gradient relativeL2 errors, not training-quality metrics.
[Aggregate receipts](../results/trained_gradient_diagnosis_20261024/controls/sdpa-fp32-reproducer-report.json)
contain forward and backward measurements for every case. No private model,
source text or activations are needed to reproduce them.

In the separate full trained-model control, native FP32 global gradient error
falls from0.029312 to0.00012068 when the NKI attention route is disabled. All
256 natural TRAIN windows, weights and target normalization remain identical.
This is a practical workaround for FP32 qualification. BF16 full-model error
remains large, so this finding does not explain that entire discrepancy or
retroactively invalidate all fitted-model comparisons. Isolated primitive
RMSNorm, linear, SwiGLU and SDPA fixtures are numerically healthy; the current
evidence does not establish a separate BF16 operator bug.

A useful contribution would make SDPA route/precision selection explicit,
respect a requested FP32 accuracy policy, or document and warn about the
automatic downcast. A gating change must also validate backward selection,
cache policy and performance. This is a local reproducer and contribution
candidate only. No upstream issue, message or PR has been sent.


### Prospective precision optimization, not a qualified replacement

The CPU-only FP32-head/BF16-body screen still misses the registered2% gradient
criterion (3.37% on eight fixed TRAIN windows), so the active execution stays
FP32. [TorchNeuron's official type documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/about-pytorch-neuron.html)
also advertises FP16 support, while [its best-practice guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/reference/best-practices-pytorch-neuron.html)
recommends BF16. FP16 with explicit loss scaling could be a bounded future
CPU-gradient screen; actual operator accuracy, underflow/overflow, native support,
complete-state recovery and throughput are unqualified. No FP16 model update,
backend change or new qualification is launched by this note.

## October24: Muon BF16 compiled/eager numerical discrepancy

The new Parcae optimizer fails its registered actual-update gate before any
scientific use. Compiled BF16 Muon differs from independent CPU updates by
10.26–10.44%; eager BF16 reduces that to0.94–2.32% but still misses the unchanged
2% gate on the signed third fixture. AdamW agrees and every replica is identical.
The large embedding update would hide this in an aggregate-only error check.

The [checkpoint-free stage reproducer](../scripts/repro_muon_polar_precision.py)
compares same-version Torch2.14 CPU eager, native eager and native compiled
outputs for four matrix geometries. Momentum agrees exactly. BF16 compiled
normalization differs before the polynomial loop; the difference grows through
the loop. Diagnostic FP32 polar outputs agree within0.011% in these fixtures.
No unexpected fallback occurs. These are numerical arrays, not a research corpus.
[Stage evidence](../results/muon_polar_diagnosis_20261024/REPORT.md) ·
[Complete-update failure](../results/parcae_early_optimizer_20261024/REPORT.md) ·
[Eager failure](../results/parcae_eager_optimizer_20261024/REPORT.md).

This identifies a reproducible execution-mode difference, not its compiler-pass
cause or an established PyTorch contract violation. A potential contribution is
clearer compiled BF16 accuracy guidance or a precision-preserving option for
sensitive polynomial optimizers. No SDK patch, upstream message, issue or PR has
been submitted. A prospective FP32-polar optimizer is an explicitly different
recipe and must pass independent update/recovery/performance gates before use.
# Math2 attention backward: CPU arithmetic reproduction,October25

The installed native wheel's `torch_neuronx/_sdpa_math.py` has a reproducible
CPU arithmetic discrepancy against PyTorch's MATH reference on FP16 causal
self-attention arrays. The formal fixtures are numerical software inputs,
not generated research examples. No installed package is patched and nothing
is submitted upstream. Actual compiled behavior remains a separate gate.

With6heads,128-dimensional normalized identical query/key vectors and32
positions,the forward relative error is0.0316%,while query/key gradient errors
are67.1%/61.4%. At1024 positions they are2.295%/2.090%. The report includes
absolute errors and reference norms;these are deliberately sensitive fixtures,
not a claim that trained queries and keys are identical.

The shared backward formula adds a correction based on the difference between
the saved output and a recomputed local output. Its forward has already rounded
the saved output for FP16 inputs. Removing that rounded-output contribution
diagnostically sharply reduces query/key error. Calling attention with FP32
inputs,then casting its external output back,preserves the internal output
through backward and avoids editing or copying SDK formulas.

The explicit-FP32 call reduces the32-position CPU query/key errors to0.0441%/
0.0414%,and the1024-position errors to0.00823%/0.00796%. The causal model wrapper
retains every original parameter,state name and RNG draw. It remains unqualified
for native training:the queued check must reproduce actual compiled behavior,
audit all32 fixture replicas,and compare complete512-window gradients at both
mature checkpoints with unchanged2%/0.003-nat thresholds. Optimizer/update/
recovery/throughput qualification is still required before scientific adoption.

[PyTorch documents FP32 math-SDPA intermediates for half inputs](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html).
That reference explains the comparison;it does not prove this is the cause of
any language-model result. The existing full-model global-gradient checks passed
their declared gates and all fitted-model/efficiency findings remain reported.

[CPU reproducer](../scripts/repro_sdpa_math2_cpu.py) ·
[Exact source digest and fixtures](../results/sdpa_math2_investigation_20261025/cpu-v1/plan.json) ·
[CPU result](../results/sdpa_math2_investigation_20261025/cpu-v1/report.json) ·
[Bounded native qualification](../results/sdpa_math2_investigation_20261025/native-v1/plan.json).

## SDPA Math2 saturated-gradient fixture,October25

Installed torch-neuronx2.14.3.0.282+dc70ffa,with NKI SDPA disabled. The
installed `_sdpa_math.py` hash and exact numerical-array digests are public in
results/sdpa_math2_investigation_20261025/cpu-v1. The CPU explicit backward
differs from autograd through its own identical forward:query/key relative
errors67.1%/61.4% on32-position RMS-normalized Q=K FP16 arrays. Output
rounding contributes to the backward correction;close scalar outputs alone
do not qualify the derivative. This is a software fixture,not research data.

Actual compiled32-rank capture reproduces query/key48.4%/44.3% relative error
on the same fixture,with0.0316% forward error and identical packets on all
replicas. A cast-only local workaround which passed CPU still gives37.2%/34.0%
on device,and fails the fixed2% gradient gate. FP32-input control cases pass.
No installed SDK patch,scientific update or upstream message was made. The
first manifest-packaging failure and subsequent numerical failure are retained
in separate versioned execution receipts. A differentiable FP32 primitive
fallback is separately registered for qualification;no speed or training
benefit is established. See native-v2/operator-audit.json and the adjoint
CPU report. Full-model errors and the scientific ranking cannot be inferred
from these operator relative errors.

The subsequent eight-case native primitive route passes on all32 replicas and
both exact-input mature full-batch audits. A standalone CPU reproducer and a
local review draft now isolate the saved-output correction:replacing only
that Python-backward input with recomputed unrounded output changes saturated
query error67.09%→0.0743%. This counterfactual is not a native patch. See
[review draft](OSS_SDPA_MATH2_REVIEW_DRAFT_20261025.md). No upstream submission.
# October27–28: stable target loss and bounded checkpoint staging

The workspace port exposes two independently reproduced conditioning/resource
issues. Neither establishes a vendor defect or explains earlier model rankings.

At extreme prediction confidence,standard FP32 target-gradient subtraction can
lose substantial relative precision on CPU as well as Native. The mathematically
equivalent target-relative form in `compute_trainium/stable_target_loss.py`
uses softplus(logsumexp of non-target logits minus the target logit). Independent
FP64 probability-gradient fixtures include100,352-way outputs. The all64-rank
Native fixture's maximum relative gradient error is2.32458e-6;the actual1.485B
fresh-source mature audit's largest named error is6.68379e-6 against an eager
HF-body/FP64-loss reference. The failed older saturated history is retained;
fixed-weight success does not repair that trajectory.

Reproducers and receipts:
`tests/test_stable_target_loss.py`,
`scripts/qualify_stable_target_loss.py`,
`results/workspace_loss_stability_20261027`,
`results/workspace_stable_loss_20261027`,and
`results/workspace_online_mature_audit_20261027/recovery-v2`.

With a192GiB worker,64 simultaneous full-state staging operations can exceed
the allowance even after neural updates finish. A separate zero-update check
also OOMs while retaining whole-matrix FP64 gradient-comparison temporaries.
The revised path streams comparison slices,releases compiled call wrappers
after the final backward,and writes bounded rank waves. At actual model size,
eight concurrent writers preserve all895 actor/scorer/AdamW tensors and RNG
exactly;the revised runtime-input gradient matches the prior qualified Native
gradient exactly. Completed files are flushed and their file cache advised
away before later waves. CPU four-process replay covers the complete state.
The all64-rank scientific replay remains a separate execution outcome.

See `scripts/qualify_workspace_checkpoint_waves.py`,
`compute_trainium/workspace_native_state.py`,and
`results/workspace_checkpoint_waves_20261027/recovery-v2`.
New code also restores the already qualified CPU-tensor-to-device pattern for
runtime scalar inputs. The memory contribution of direct-device scalar
construction has not been isolated;do not call it the established OOM cause.

## October28: persisted HLO references transient NKI binaries

After cache maintenance and a fresh worker,qualification stops at its first forward.Eager index_select HLO references stride2_gather .colz files below the prior container-local /var/tmp/nki-intermediate-cache;compiler exits70 with NCC_EVRF059.The supervisor fails closed,zero recorded completed updates,no OOM.Installed hlo_cache.py exposes separate TORCH_NEURONX_HLO_CACHE and TORCH_NEURONX_HLO_COMPILE_CACHE switches;NKI compiler/_disk_cache.py exposes NKI_COMPILE_CACHE_URL.The bounded recovery bypasses both persistent HLO modes and uses a persistent NKI directory,then rechecks all179 gradients before fitting.This is a local cache-dependency failure,not a scientific numerical result.No upstream submission.Private logs and SDK sources remain local.


## October28: unresolved1024-position cached-forward numerical discrepancy

A new zero-update dependency evaluator stops before panel scores at its independent full-size HF reference gate. Both Native attempts use the qualified FP32 model/scientific states but a new1024-position causal prefix and16 independent cached queries. Rank-max logit discrepancies sometimes exceed.001 while rank0 real-query comparisons are much smaller;one diagnosis also exceeds.0002 real-target NLL on an embedding-replay identity. A separate diagnosis records rank-max and rank0/per-query errors before assertions;it is not a numerical pass. The first worker did not retain error magnitudes before assertion,an evidence limitation retained rather than filled in.

The same cached CPU implementation agrees with independent HF dense2D references. Removing only fully masked future KV passes CPU output/all-gradient equivalence but does not qualify the Native retry. No SDK defect,rounding-mode cause or replica corruption is established. No precision/environment/gate relaxation or third Native attempt follows. Both stopped costs and the separate diagnosis remain. The scientific panel instead runs on the CPU graph after all12 same-tolerance independent reference checks. These failures concern the new inference geometry and do not invalidate the earlier160-position training audits. Reproduction scripts: `scripts/run_dependency_forward_diagnosis.py`, `scripts/evaluate_recoverable_dependencies.py`;small aggregate measurements are in `results/recoverable_dependencies_20261028`. Model/source fixtures stay private;no upstream submission.
