# Complete low-depth training/profile engine, revision3

Same original three-arm initialization, natural source/depth/noise tables,
optimizer arithmetic, context1024, accumulation8/16 and100-warm-update gates.
This host-only revision changes the preceding recovery receipt, caps each owned
worker at128GiB and returns freed CPU comparison arenas between phases. All48
source hashes are frozen. The model trainer and independent CPU audit source
from revision2 are unchanged. Stopped predecessors and their costs are retained.

FP32 native complete gradients and explicit FP32-polar optimizer/recovery have
passed. Include FP16 only after its separate native gradient pass; otherwise
qualify FP32 alone. Exact complete state/RNG on load, per-family first-update
and trained-global-gradient checks, replica equality, numerical next-update
replay,100 warm end-to-end steps and90-minute whole execution cap remain.
No scientific fit, held-out scoring or automatic large-run launch.

[Plan](native-plan.json) · [Current status](native-status.json) ·
[Original engine protocol](../../../docs/PARCAE_TRAINING_QUALIFICATION_20261024.md) ·
[Host admission correction](../../../docs/PARCAE_RECOVERY_ADMISSION_20261024.md).
