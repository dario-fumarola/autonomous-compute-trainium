# Complete low-depth training qualification

All10 registered cases pass. These are numerical/throughput fixtures using
natural TRAIN windows, not held-out learning curves or scientific fits.

Each case executes20 warmup updates,100 timed warm updates, an uninterrupted
121st update, and a fresh-process replay from the complete120-update state.
The complete state loads exactly; all32 replicas agree. Independent pinned
upstream FP32 scoring differentiates the complete next global objective at
the trained checkpoint using the exact supplied hidden states and sampled
depths. Overall and named-family gradient comparisons pass the unchanged2%
gate. The first actual native update is checked separately against independently
clipped gradients and the released optimizer with its sole polar dtype amendment.

| Arm | Accumulation | FP32 targets/s | FP16 targets/s |
| --- | ---: | ---: | ---: |
| Ordinary | 16 | 424,852 | 479,674 |
| True one-pass Parcae | 16 | 363,076 | 435,366 |
| Sampled lambda2/minimum1/tail1 | 16 | 326,499 | 394,766 |
| True one-pass Parcae | 8 | 332,620 | 383,408 |
| Sampled lambda2/minimum1/tail1 | 8 | 299,621 | 349,724 |

Rates are useful original targets per second per eight-chip/32-rank job,
including the timed training objective, accumulation, clipping and optimizer.
Each timed interval contains100 updates. Cold launch, compilation, checkpoints,
CPU audit and recovery are charged separately; these are not end-to-end campaign
rates. The common accumulation16/global512 geometry wins in every measured pair.

Use FP16 body/FP32 head and master parameters, manual8192 scaling/unscale, eager
FP32-polar Muon and fused unscaled AdamW. This policy is independently qualified
and shared across all scientific arms. It is not an exact release reproduction:
the rejected BF16-polar implementations and their failure receipts remain public.

Active execution took2486.839seconds. Two complete predecessor warm profiles
were reused by checksum rather than repeated. Their original costs remain in
attempt3; this attempt additionally charges recovery and corrected independent
audits. The earlier checker failed on parameter registration order, despite equal
named optimizer memberships; the repair validates the complete independent
membership policy before aligning by name. No optimizer arithmetic or gate changed.

[Exact results](report.json) · [Costs](execution-costs.json) ·
[Frozen plan](native-plan.json) ·
[Checker repair](../../../docs/PARCAE_PROFILE_AUDIT_REPAIR_20261024.md).
