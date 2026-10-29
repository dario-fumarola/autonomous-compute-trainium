# Tied embedding-role gradient diagnosis

The observed role gradients are nearly orthogonal,not opposing. This diagnostic
does **not** support cancellation as the source of the fixed-one model's deficit.

| Model | Input/output cosine | Input/proposal cosine | Output/proposal cosine | Reconstruction relative error |
| --- | ---: | ---: | ---: | ---: |
| Fixed-one | +0.013808 | +0.041143 | +0.000548 | 4.82e-6 |
| Ordinary equal-data | +0.015787 | — | — | 4.30e-6 |

Identical leaf copies split input lookup,output projection and proposal-value
roles while preserving the forward exactly. Their summed gradient reconstructs
the original audited tied-table gradient. The norm of a vector sum below the
sum of norms is expected for orthogonal vectors and is not evidence of conflict.

Both models use the same256-window/130,860-target TRAIN audit batch. No weights
are updated,no evaluation sources are read,and no untied fit follows. This one
batch cannot exclude optimization conflicts elsewhere in training. The measured
fixed-versus-sampled training difference motivates a separately registered
budget-conditioning test rather than claiming a diagnosed parameter-sharing cause.

CPU time192.42seconds;peak RSS5.48GiB;zero assigned chips.
[Registration](plan.json) · [Aggregates](report.json).
