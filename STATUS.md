# Status — 2026-10-22

The fresh natural-text experiment is built and qualified. No large scientific run has launched. Read the [qualification report](results/qualification_20261022/REPORT.md), [protocol](docs/PROTOCOL.md), and [proposed budget](results/qualification_20261022/budget.json).

Fourteen CPU tests pass. Random-initialized 150,436,608-parameter TorchTitan models pass ordinary, early-pause and late-pause native update/recovery checks. CPU gradient agreement is checked for ordinary and early pause. Both whole-instance training benchmarks use all 64 logical cores across all 16 Trainium2 chips and finish with identical full model/optimizer state on every replica.

| Measured warm training | Real targets/sec |
| --- | ---: |
| Ordinary | 158,031 |
| 16 pauses per 128 input positions | 112,575 |

Separate single-core inference checks agree with CPU and preserve weights. Teacher-forced window latency is 9.18 ms ordinary versus 33.18 ms with pauses. These are not cached autoregressive decode measurements. The small three-arm natural-text learning diagnostic improves fresh DEV NLL from about 11.54 to 8.09, but establishes no pause benefit. TEST remains unevaluated.

The proposed 3B-target, three-arm, three-seed pilot plus separate LR sweeps budgets 118–177 node-hours with a 2–3× allowance. It remains undertrained by design; a subsequent paired continuation to 30B targets is priced separately. Before scientific launch: freeze a larger natural corpus and order, audit duplicates, and perform the short full-topology late-switch/memory-stability check. Its throughput is currently estimated from the two measured mechanisms.

No accelerator workers remain active at handoff. The development container is stopped after qualification; restart it with `docker start autonomous-compute-neuron233` before a new bounded check. All sources, raw data, logs, states and caches remain under `/run/user/1000/autonomous-compute`. The wrapper snapshots its own launcher and the source for each attempt, records exit codes and costs, and checks the actual container limit plus a 512 GiB host reserve.

Failures are preserved in the [resource ledger](results/qualification_20261022/resource-ledger.json). Two 256 GiB pause compiles and a combined training/inference attempt at 416 GiB exceeded the container limit. The completed pause benchmark passed hardware checks before a post-success launcher error; its original exit 127 is retained. The immutable launcher subsequently completes ordinary training and separate inference with exit 0. See [OSS notes](docs/OSS_NOTES.md) for the reproduced compiler-error propagation defect, device-index workaround, and memory observations. No upstream submission was sent.

The predecessor pilot is archived. Its five stopped qualifications used 11.8631 assigned chip-hours. The original 1.7B numerical results remain frozen: 2,516 protected files match commit d07d89e. The private meeting transcript is not published. Training and evaluation source data contain no generated arithmetic, synthetic rewrites, or teacher traces.
