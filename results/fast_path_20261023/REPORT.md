# Post-science software and throughput qualification

The scientific comparison and its twelve-model TEST confirmation closed first.
These are numerical and performance checks, using natural TRAIN windows and
software fixtures. They do not change completed scientific models.

Unscaled fused AdamW passes every distinct 150M weight shape, including the
32000 by 768 vocabulary matrix, irregular/offset views, changing learning rates,
FP32 moments and exact optimizer continuation. Maximum CPU parameter error is
**1.56e-07**. The 32-rank accumulated global gradient matches CPU to relative L2 **0.00236**.

Two jobs use disjoint eight-chip halves of the same node, after a shared warm
start. Their global updates each contain 256 source windows, matching the
original 64-rank/batch 2/accumulation 2 global batch. Every full replica and both
fresh-process optimizer recoveries match exactly.

| Path | Chips | Ranks | Local batch | Accumulation | Warm updates | Real targets/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary | 8 | 32 | 4 | 2 | 100 | 873,844 |
| Shared two-pass | 8 | 32 | 4 | 2 | 100 | 576,011 |

The ordinary warm interval overlaps shared training for approximately 100%; the shared interval overlaps ordinary for 65.9%. The equal-update warm pair completes 52,337,076 real targets in 45.43 seconds, or 1,152,021 targets/sec across the pair. Summing the two individual average rates would overstate this paired completion rate.

This jointly changes optimizer implementation, local batch and DDP topology;
it does not isolate one cause. Cold setup, final checkpoint hashing and future
periodic scientific checkpoint overhead are separate from warm rates. No new
large training campaign is launched from this qualification.

A separate, bounded readout check uses the same ordinary checkpoint and all
614 DEV documents / 777,348 real targets. All three NLL values are exactly equal.

| Readout ranks | Assigned chips | End-to-end native-launch seconds | NLL |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 63.78 | 2.518071 |
| 16 | 16 | 58.64 | 2.518071 |
| 64 | 16 | 85.90 | 2.518071 |

Sixteen ranks use one logical core per physical chip. These are single panel
timings with different cache histories, not a steady inference or generation
benchmark. All checks enforce unchanged weights and complete target coverage.
The completed scientific confirmation retains its qualified 64-rank execution.

The optional direct ATen AMP-metadata fixture reproduces a separate mismatch:
normal unscaled updates match CPU, but provided overflow/unscale arguments are
ignored. High-level GradScaler reachability is untested. Scientific training
uses plain AdamW; the qualified candidate uses unscaled BF16 with FP32 masters.
Neither uses those optional arguments. See the reproducible contribution note.

[Detailed summary](summary.json) · [Native qualification](qualification.json) ·
[Optional-metadata contract](amp-metadata-contract.json) ·
[Contribution note](../../docs/OSS_FUSED_ADAMW_METADATA_20261023.md).
