# PyTorch and Trainium observations

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
