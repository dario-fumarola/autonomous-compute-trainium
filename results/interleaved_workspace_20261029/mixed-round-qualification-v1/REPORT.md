# Alternating numerical-depth training qualifies

Both residentK2/K3 graphs pass cold independent CPU gradients. The100 warm
updates average0.283439seconds, with50 at each depth:0.266878s forK2 and0.299999s
forK3. The51,747,328-parameter graph uses64 ranks across all16 chips, batch2/rank.

Fresh-process update104 atK3 exactly reproduces the complete model/optimizer/RNG
state. All replicas agree. The independent128-window global mature audit
passes: largest gradient-family relativeL2 error1.55993e-6, well below the
unchanged0.02 threshold. There is no scientific quality score or retained
scientific initialization from this disposable qualification.

The first controller completed warm training, then stopped before replay
admission when available host memory stayed below the protected512+192GiB
allowance for180seconds. A lossless, hash-verified relocation of one closed old
qualification array out of tmpfs restored headroom. A separately recorded
recovery ran only the unstarted replay and audit, retaining the original plan,
weights, gates and absolute deadline. No reserve was relaxed. The first stop,
assigned work and reservation opportunity cost remain in the ledger.

[Final gate receipt](report.json) · [Original costs](execution-costs.json) ·
[Recovery and costs](recovery-v1/report.json) ·
[Recovery ledger](recovery-v1/execution-costs.json) ·
[Original stop](recovery-v1/prior-stop.json).
