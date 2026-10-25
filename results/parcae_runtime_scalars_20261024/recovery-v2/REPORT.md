# Scheduled optimizer inputs qualified

Pass. Runtime FP32 tensor inputs replace changing scalar fills without changing
the precise Muon mathematics,moment layout or parameter grouping. The installed
native beta specializes `fill_.Scalar` on its value. The first scientific screen
observed about4.4seconds/update while values changed and1.1–1.2seconds at a
constant schedule; it was stopped before any held-out scoring rather than
extending its frozen budget.

| Arm |100-update scheduled targets/s per8chips|CPU raw gradient relative error|
|---|---:|---:|
|Ordinary|491,376|0.0213%|
|True one-pass|420,279|0.1085%|
|Sampled lambda2/minimum1/tail1|399,894|0.0773%|

Every100-update interval changes LR and decay on every step. Exact complete
state/RNG loading,32-replica equality,prescribed first actual updates,independent
trained global gradients and recovered global gradients pass. Recovered versus
uninterrupted parameter-update errors are below9e-6 relative in this fixture;
this does not assert bit-identical complete future trajectories.

Recovery-v2 retains the four completed ordinary/recurrent native phases by
checksums. It executes only the missing one-pass warm/replay pair and reruns
the interrupted CPU audits in new output directories. The preceding controller
stopped because another approved queue claimed native resources during a gap;
explicit predecessor phase closures now gate admission. No data,numerical gate
or optimizer arithmetic changed in that repair.

Recovery wall time786.927seconds. The original execution,interrupted CPU work,
retained phase costs and additional work remain separately recorded. None of
these qualification updates are scientific training exposure.

[Exact results](report.json) · [Costs](execution-costs.json) ·
[Phase-recovery policy](../../../docs/PARCAE_RUNTIME_PHASE_RECOVERY_20261024.md).
