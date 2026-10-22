# Natural-text qualification — 2026-10-22

The new natural-text experiment is implemented and qualified at 150,436,608 parameters. This establishes correct execution and a measured budget. It does not establish a benefit from pauses.

## Data and model

The qualification corpus contains 1,800 natural source documents: 600 educational web, 600 mathematical web and 600 repository code files. Immutable source revisions and file hashes are in [the manifest](data-manifest.json). Globally grouped host/repository splits contain 1,998,577 TRAIN, 307,646 DEV and 279,585 TEST real tokens. TEST has not been evaluated. Source content, token arrays and the private meeting transcript are absent from this repository. The current convenience sample needs a larger multi-shard corpus and near-duplicate audit before scientific training.

The model uses the pinned Mistral 32k tokenizer, random weights, 20 Llama-style layers, width 768, FFN width 2048, 12 query heads and 4 KV heads. Input/output weights are tied. Their 24.576M parameters are included in the total, and the output projection is included in FLOPs. All three arm qualification runs share the same initial state digest. No teacher model, generated arithmetic, synthetic rewrite corpus or RL was used.

## Correctness evidence

Fourteen CPU tests cover source isolation, real-target coverage, causal masking, pause gradients, full-vocabulary loss and gradients, optimizer continuity, paired statistics, budgets, and exact static-slice/gather equivalence. The native checks use TorchTitan models from the pinned Neuron fork on the native PyTorch backend.

- Ordinary, early pause and late pause each complete actual 150M AdamW updates on hardware and replay the full saved model/optimizer state exactly. The early-pause gradient relative L2 error against CPU is approximately 1.0e-6; the ordinary check is approximately 0.0022.
- A 64-rank small-model global-batch gradient check agrees with CPU to relative L2 error 0.00054. This verifies one distributed gradient reduction, rather than a duplicated average.
- Both 150M ordinary and pause whole-instance benchmarks finish with identical full model and optimizer digests on all 64 replicas. All 16 physical chips are used.
- Separate inference checks agree with CPU for both mechanisms, leave all model weights unchanged, and record no fallback operations. DDP records `aten::_to_copy`; we do not claim zero distributed fallbacks.

The late-arm correctness/recovery check is single-core; the whole-instance throughput estimate for Late combines the two measured mechanisms. A long-lived whole-instance late switch still needs a short production stress check, particularly because retaining multiple compiled graphs is expensive.

## Measured performance

FP32, 64 logical cores across 16 Trainium2 chips, local batch 1, 512 real targets per rank, 32,768 real targets per optimizer update. The pause arm processes 576 input positions per rank. Warm training includes data transfer, backward, native AdamW and synchronized DDP. The steady windows are short: 12 updates each.

| Measurement | Ordinary | 16 pauses / 128 input positions |
| --- | ---: | ---: |
| Whole-instance warm real targets/sec | 158,031 | 112,575 |
| Cold first training update | 72.1 s | 172.3 s |
| Single-core teacher-forced inference latency | 9.18 ms | 33.18 ms |
| Single-core teacher-forced inference real targets/sec | 55,783 | 15,431 |

The pause arm costs approximately 1.40× the measured training time and 3.62× the measured inference time. The dense-matmul FLOP ratio is only 1.1198×. These are different quantities; the FLOP ledger is an estimate and excludes elementwise/optimizer operations. Inference here scores a preloaded real-token window, not KV-cached autoregressive generation. No node-level inference rate is extrapolated.

The isolated Ubuntu 26.04 userspace uses PyTorch 2.14.0+cpu, torch-neuronx 2.14.3.0.282+dc70ffa and compiler 2.28.7631.0+8f7b0cf5. The host driver and original XLA environment are unchanged. Driver DMA fallbacks are included in the measurements. See [setup](../../docs/NATIVE_SETUP.md).

## Learning diagnostic

A separate 4.39M CPU model receives 64 updates and exactly 59,248 real targets per arm, using identical initial weights and natural training batches. Fresh DEV is a fixed 12-document convenience panel from held-out source groups.

| Arm | DEV NLL before | DEV NLL after |
| --- | ---: | ---: |
| Ordinary | 11.5439 | 8.0919 |
| Early pause | 11.5416 | 8.0981 |
| Late pause | 11.5439 | 8.0909 |

This shows the pipeline can learn natural text and improve on fresh documents. It is not a scientific comparison, a converged model, a seed-variance estimate, or evidence that pauses help. See [the complete diagnostic](natural-learning-diagnostic.json).

## Proposed budget and launch gates

The [budget](budget.json) proposes three paired seeds and three arms at approximately 3B real targets per fit. This is about 20 targets per parameter and remains a signal-seeking pilot. Warm training totals 57.0 node-hours, plus 1.9 hours for nine 100M-token LR candidates. The 2–3× planning range is **118–177 node-hours (4.9–7.4 days)**, or 1,886–2,829 physical chip-hours. Evaluation, preparation, checkpoint and sustained-rate uncertainty sit within that explicit allowance; it is not a measured sustained-rate guarantee.

A subsequent ordinary/early paired seed continued from 3B to 30B targets adds about 114 warm node-hours, or 228–342 planning hours. Its purpose is to test whether a gain survives later training, not to declare convergence from a single paired seed. Additional sizes and independently tuned FLOP-budget endpoints are outside the first budget.

Predeclared equal-estimated-FLOP curve milestones are 3.000B ordinary, 2.679B early and 2.839B late real targets. These use a common source-token training schedule; they are not separately optimized endpoint runs. Report token-matched and estimated-FLOP-matched curves separately. Validate or bound the omitted FLOP terms before a strong compute-efficiency claim.

No large scientific run has launched. Next: freeze a larger natural corpus and its order, audit duplicates, run a short full-topology late-switch and memory-stability check, then execute the separately budgeted LR sweeps. A 3B negative cannot substitute for the planned longer continuation.

## Failures, preservation and OSS evidence

All stopped/failed attempts remain in [the resource ledger](resource-ledger.json), including two device-index failures, two 256 GiB compilation OOMs, the combined training/inference OOM at 416 GiB, and the post-success shell-wrapper error. Uninstrumented setup mini-checks are marked as such, not charged as zero. The inclusive elapsed whole-node window covers setup, code work, gaps and overlaps. Historical pilot costs remain a separate 11.8631 assigned chip-hours.

The pause hardware report is valid despite its wrapper exit 127: it completed all updates and replica checks before the shell error. The fixed immutable launcher subsequently completes the ordinary and inference runs with exit 0. See the original report and closure rather than treating the wrapper failure as a scientific failure or a clean exit.

The predecessor [preservation audit](predecessor-preservation.json) verifies 2,516 protected numerical/source/configuration/result files against commit d07d89e with no changes. Its non-numerical train/README.md is outside this audit. The old public pilot is archived, and this repository has an independent history.

A minimal native compiler-failure reproducer demonstrates an incorrect returned tensor without an exception after an intentional compiler error. The [reproducer result](compiler-error-repro.json) and [OSS notes](../../docs/OSS_NOTES.md) retain this contribution candidate. No upstream issue, PR or message was submitted.
