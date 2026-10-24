# Actual-shape Parcae optimizer preparation

**Independent CPU update vectors pass; native execution is queued.**
This is numerical preparation, not model learning or a scientific advantage.

The ordinary140,781,888-parameter and recurrent141,963,840-parameter points use
the existing qualification weights and complete256-window natural-TRAIN
gradient vectors. Three prescribed signed/scaled-gradient updates exercise
changing rates, momentum, cautious Muon decay, and full named moment state.
The independent expected algorithm is the pinned released MuonAdamW; the
candidate uses fused CPU AdamW and the explicit Muon implementation.

Maximum update error relative to the reference update is1.65204e-6; maximum
complete-moment relative error is4.72221e-8. CPU wall time through the terminal
report is124.485s, with zero chips and zero research updates. Each implementation
does three prescribed software optimizer steps per arm. These gradients are
fixtures, not gradients recomputed after updating the model.

[CPU vectors and errors](attempt2/cpu-report.json) ·
[Preparation plan](attempt2/preparation-plan.json) ·
[CPU cost receipt](attempt2/cpu-execution-costs.json).

The first attempt failed before any update because its public report directory
was missing. It remains preserved with its source hashes and zero-chip cost.
[First failure](attempt1/failure.json).

Native execution revision2 waits for the separate global native gradient gate
to pass and for managed workers to release all chips. It then uses two
eight-chip halves for actual-shape prescribed optimizer updates and a fresh-
process full-state restore/next update. Update and moment gates apply separately
to Muon and AdamW as well as overall. Exact state on load and complete replica
agreement are required. Latest phase start02:00UTC October25; thirty-minute
execution limit. Waiting reserves zero chips. No fresh panel or terminal alpha.

Prelaunch inspection found that the native capture included persistent buffers
and a duplicate tied-head alias in its comparison vectors, while the CPU vectors
contained named parameters only. The waiting-only revision1 was canceled after
45s, with no native attempt or update. Revision2 corrects the vector inventory;
a regression fixture contains both a buffer and tied alias. All CPU vectors,
the model, and optimizer algorithm remain unchanged. Original plan and both
unstarted source files remain available in the cancellation archive.
[Unstarted revision1](attempt2/unlaunched-r1/cancellation.json) ·
[Registered revision2](attempt2/native-plan.json) ·
[Native protocol](../../docs/PARCAE_NATIVE_OPTIMIZER_PROTOCOL_20261024.md).

Native updates, sampled-depth forward/backward training, clipping, full trainer
recovery and100 warm training updates still need qualification. No scientific
fit or measured training budget is launched by this preparation.
