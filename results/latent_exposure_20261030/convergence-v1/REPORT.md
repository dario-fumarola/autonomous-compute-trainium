# Exact prediction is stable; the original K2 distribution guard narrowly fails

The unchanged six-TRAIN-window diagnostic completes on the audited billion-target
latent model. The exact sequential result remains the published quality endpoint.
This check does not change any model, held-out score or advancement decision.

| Domain | K2 minus exact NLL | KL(K2 to exact) | Original domain guard |
|---|---:|---:|---|
| Web | +0.000645 | 0.010705 | **Fail** |
| Math | +0.003177 | 0.009935 | Pass |
| Code | +0.006087 | 0.006915 | Pass |

The original distribution tolerance is 0.01, so the all-domain K2 eligibility
check fails. Do not relax it after seeing the result. K3-to-exact KL is
0.002527 / 0.002553 / 0.000947, respectively. K16/K32 agreement is better than
5.05e-10 in every domain. These are numerical refinements of one generated
state per source token, not supported inference thought budgets.

The larger cheaper-path errors matter: K1 minus exact NLL is +0.033935 on web,
+0.031507 on math and +0.031766 on code. Deleting solver work from this saved
recipe cannot be assumed harmless. A separately trained cheaper approximation
would be a new optimization experiment, with no result implied here.

The cached/full-prefix maximum target-loss discrepancy is 6.41e-6, within the
unchanged 0.0002 reference tolerance. As in the original diagnostic, the body
uses FP32 and loss/KL arithmetic FP64. This differs from the full FP64 scientific
quality readout; these six TRAIN windows do not replace that readout. They are
outside the complete prospective training prefix and are not an independent
evaluation panel.

Runtime is 39.496 wall seconds, 16 CPU threads, 1.910 GB peak RSS, zero chips
and zero optimizer updates. It overlaps the active ordinary control; its host
work is recorded separately and reservation opportunity time is not added twice.
No fresh reserve, terminal alpha, new fit or automatic promotion is admitted.

[Full result](report.json) · [Registration](plan.json) ·
[Protocol](../../../docs/LATENT_EXPOSURE_CONVERGENCE_20261030.md).
