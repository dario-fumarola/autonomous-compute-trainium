# Mature cheap-control precision diagnosis

The failed scientific comparison remains stopped. This separately frozen
45-minute diagnostic uses its exact checkpoint and next 512 TRAIN windows,
with zero optimizer updates. Initial native variants move only the adapter to
FP32 or execute the full body in FP32; a prespecified readout-plus-adapter FP32
variant is conditional on the first failing and the full-FP32 reference passing.
The original 2% gradient-family and 0.003-nat loss gates remain unchanged.

The one-window CPU implementation screen passes: all three branches with
body autocast disabled give exactly the same loss and gradients as the existing
FP32 implementation. It uses captured native latent states, avoiding cross-
PyTorch random-number differences. Preparation took 42.796054 seconds. This
admission evidence is not native qualification or a learning result.

Independent reference computation covers the complete 512-window objective.
The same pinned FP32 vector may be reused only with exact point, state, source
and count identities. No trained weights, source groups or quality targets are
changed. No TEST, reserve access, terminal alpha or scientific fit is admitted.

[Plan](plan.json) · [Protocol](../../docs/MIXING_PRECISION_DIAGNOSIS_20261026.md) ·
[Preparation attempts](preparation-attempts.json).
