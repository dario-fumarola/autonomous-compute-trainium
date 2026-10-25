# Verified CPU-only fixed-depth curve preparation

Original random seed5231 weights, original authentic-source packing and the
complete natural-source/auxiliary/work counters verify. Every extended anchor
count in the old provisioned prefix is identical. Both new depth/work arrays
and all saved counters read back exactly. No research source is generated and
no initializer,source or old checkpoint is changed.

| Endpoint | Updates | Real source targets | Complete nominal matrix work |
| --- | ---: | ---: | ---: |
| one-1b | 1,920 | 1,004,887,590 | 5.282838e+17 |
| fixed-1b | 1,920 | 1,004,887,590 | 6.363450e+17 |
| one-work-1b | 2,312 | 1,210,049,128 | 6.361418e+17 |
| one-2b | 3,840 | 2,009,762,680 | 1.056568e+18 |
| fixed-2b | 3,840 | 2,009,762,680 | 1.272690e+18 |
| one-work-2b | 4,625 | 2,420,609,556 | 1.272559e+18 |

One's unused fraction of each fixed-two budget is
0.031935% at1B and
0.010316% at2B;the next complete
update exceeds each budget. Every endpoint has its own full cooldown. These
are nominal matrix counts,including the common zero-weight training-only head
and all32 replicas' optimizer matrices;not matched chip-hours.

CPU preparation takes39.16s,zero assigned chips and zero
learning updates. The source counter tables provision2,621,952
windows without wrapping. No DEV/TEST scoring. **Scientific launch remains
false** until the complete bounded TUNE and a separately frozen measured MAIN
plan pass. [Exact manifest](manifest.json) · [Scope](../../../docs/FIXED_TWO_SMALL_CORE_MAIN_20261025.md).

The preparation draft incorrectly describes a 10% LR floor. The unchanged,
qualified runtime actually cools LR and Muon cautious decay to zero,as its
executed TUNE receipts show. No model update was admitted by this preparation.
The separately frozen [MAIN admission protocol](../../../docs/FIXED_TWO_SMALL_CORE_MAIN_ADMISSION_20261025.md)
corrects that description before scientific admission;original preparation
source hashes and artifacts remain preserved.
