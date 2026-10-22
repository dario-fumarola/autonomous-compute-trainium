# Status — 2026-10-22

The bounded performance/trainer pass is complete. **No large scientific fit or LR sweep has launched.** Read the [new report](results/performance_20261022/REPORT.md), [budget](results/performance_20261022/budget.json), [protocol](docs/PROTOCOL.md) and [trainer details](docs/TRAINER.md).

| 150M, all 16 chips, 100 warm optimizer updates | Real targets/sec |
| --- | ---: |
| Ordinary | 855,336 |
| 16 pauses per 128 input positions | 353,879 |

The chosen configuration uses 64 ranks, context 1024, local batch 2, accumulation 2, BF16 autocast with FP32 masters/moments, token-correct DDP no_sync, packing and gradient clipping. This is 5.41× / 3.14× the original benchmark rates. The small natural qualification sample is explicitly repeated for timing; these are not scientific fits.

Twenty-three CPU tests pass. The 64-rank accumulated gradient agrees with the CPU global batch. All three tiny-model arms complete the same 196,322 real targets through the controller, including fresh-process DEV. The 150M full-topology late switch passes recovery and CPU numerical checks. Two independent 150M WSD branches finish bit-identically in full model/optimizer state, schedule, losses and mapped-data cursor; the stable checkpoint remains unchanged. Both full DEV evaluations score all 232 documents / 307,646 targets identically, with unchanged weights. TEST remains sealed. These checks establish no pause advantage.

The revised three-arm, three-seed, 3B-target pilot costs 16.77 steady node-hours including separate LR mini-fits and extra annealed FLOP endpoints. Plan **33.5–50.3 node-hours** with a 2–3× allowance, replacing the earlier 118–177 h proposal. Three billion targets remains a signal-seeking exposure; the paired 30B continuation is priced separately. Every comparison endpoint must complete its own decay.

The [new ledger](results/performance_20261022/resource-ledger.json) retains 24 attempts, including one compile OOM and one preflight rejection. Assigned accelerator time is 9.5169 chip-hours; the inclusive node window is reported separately. Sampled host MemAvailable stayed above 543.88 GiB and filesystem free space above 35.36 GiB. The supervisor checks reserves, compiler errors, OOM events and wall limits. Standalone strided-input and contiguous-control reproducers agree numerically and isolate the extra CPU round trip. See [OSS notes](docs/OSS_NOTES.md); no upstream submission was sent.

No accelerator workers remain active. The development container is stopped; models, optimizer states, raw sources and caches remain outside Git under `/run/user/1000/autonomous-compute`. Restart it before another bounded run. Plans set measured phase-specific container limits: tiny training 160 GiB, 150M training 256 GiB, DEV 64 GiB, with the separate 512 GiB host reserve.

Next scientific launch gates: freeze a larger natural corpus and 60/20/20 order, audit duplicates, build shared mapped arrays with streaming preparation, measure production I/O, then select each variant's LR at the chosen batch. Freeze rounded token/FLOP endpoints and schedule before the main fits. No synthetic research corpus, teacher traces, RL, larger model or new instance has been introduced.

The [initial qualification](results/qualification_20261022/REPORT.md) remains preserved. The predecessor is archived and its costs stay separate. The original 1.7B numerical evidence remains frozen; the private meeting transcript is not published.
