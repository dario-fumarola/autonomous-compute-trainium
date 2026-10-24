# Preserved eight-pass CPU preparation

The CPU-only preparation completes in42.854s from launch,41.419s after imports.
It saves release initialization, depth/noise tables and local numerical vectors.
No chips, optimizer updates, held-out scoring or scientific training ran.

Owner steering replaced this prospective high-depth benchmark with the
[low-depth three-arm engine](../parcae_low_depth_qualification_20261024/REPORT.md).
This directory is retained as a preparation cost and software artifact. No
eight-pass native execution plan was registered, and this is not a learning
result or a commitment to high-depth pretraining.

[CPU report](cpu-report.json) · [Preparation plan](preparation-plan.json) ·
[Costs](cpu-execution-costs.json).
