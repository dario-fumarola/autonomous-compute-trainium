# Explicit FP32-polar optimizer: arithmetic and recovery pass

Both actual-shape native primary captures pass. Maximum Muon update error is
0.13855%; complete replica states agree. The first paired recovery admission
fails the host-reserve preflight and cancels its peer; it is preserved below.
The subsequent recovery-only execution retains those exact primary states,
reduces each owned worker cap to128GiB, and **passes** for both models in92.135s.
Complete state on load is exact, replicas agree, and resumed third updates
equal the uninterrupted updates in this optimizer-only test (relative error0).
This does not promise identical complete future training trajectories.

[Recovery pass](recovery-v2/native-report.json) ·
[Recovery costs](recovery-v2/execution-costs.json) ·
[Preserved initial stop](native/native-report.json) ·
[Admission correction](../../docs/PARCAE_RECOVERY_ADMISSION_20261024.md).

Independent actual-shape CPU update vectors pass for both models and all three
prescribed steps. Maximum update error is1.652032e-6 relative to reference step
size; maximum complete-moment error is4.722248e-8. CPU time after imports is
152.374s, with zero chips/research updates. Launcher/import overhead was not
separately measured. Each arm executes three steps in each CPU implementation.

This is an explicitly different precision recipe. The independent upstream
algorithm receives exactly one audited substitution, its polar input conversion
from BF16 to FP32; decorators are removed for eager CPU reference execution.
The candidate is separately implemented. The original source, BF16 failures,
native2% family gates, drivers and SDK remain unchanged.

Native prescribed updates and fresh-process recovery use eager FP32 Muon,
fused unscaled AdamW and both32-rank/eight-chip halves. No source forward,
held-out scoring or scientific update occurs. Complete training/profile gates
remain after this optimizer pass. [CPU vectors](cpu/cpu-report.json) ·
[CPU plan](cpu/preparation-plan.json) · [Native plan](native/native-plan.json) ·
[Initial native status](native/native-status.json) ·
[Amendment](../../docs/PARCAE_PRECISE_EXECUTION_20261024.md).

Two prepare-only controller registrations initially failed to write missing
nested public directories, after writing their private plans. No hardware phase
started. Those plans and source snapshots are preserved; the intervening FP16
registration was invalidated to avoid stale prerequisite hashes. The repaired
registration passes, with the146-test suite and six targeted tests. Original
elapsed CPU time for the failed registrations was not captured.
[Unstarted optimizer registration](native/unstarted-registration/failure.json) ·
[Unstarted training registration](../parcae_low_depth_qualification_20261024/attempt2/unstarted-registration/failure.json).
