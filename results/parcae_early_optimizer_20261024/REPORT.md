# Early optimizer-only screen: Muon update gate fails

The released scientific eight-chip half ran three prescribed optimizer updates
on the recurrent141,963,840-parameter qualification point. No language-model
forward/backward, source exposure, held-out scoring or scientific update ran.
The other eight chips continued the ordinary scientific compute control.

All32 replicas have identical complete states. AdamW update relative error is
3.26e-6 to6.14e-6. **Muon update error is0.10260–0.10444**, above the unchanged
2% per-family limit. Aggregate errors are much smaller because the large
embedding AdamW update dominates them; that does not pass the Muon gate.
Muon complete-moment errors are0.00295–0.00597. First-moment inspection finds
exact agreement on step1; the discrepancy is in polar-dependent variance and
parameter updates. No recovery pass was attempted after failed arithmetic.

Worker time79.103s, assigned cost0.175785 chip-hours; controller including CPU
comparisons138.360s. These are optimizer-only timings, not training throughput.
The frozen scientific comparison and earlier CPU expected vectors are unchanged.
The unstarted full optimizer, FP16 and training-engine controllers were suspended
with zero native phases, preserving their original plans. The zero-update FP32
gradient queue remains eligible after scientific completion.

[Exact errors](report.json) · [Plan](plan.json) · [Costs](execution-costs.json) ·
[Intermediate-stage diagnosis](../muon_polar_diagnosis_20261024/REPORT.md).
