# Direct arithmetic does not give a reproducible latent-computation lead

All nine saved endpoints complete the frozen 353-source comparison. The task-specific development gate fails: only seed 730201 clears both control margins. No example, operation bin or seed is dropped after scoring.

Positive gains favor generated latent states. Each score averages cases within the three operation bins, then weights the bins equally.

| Seed | Latent answer NLL | Ordinary answer NLL | Pause answer NLL | Gain vs ordinary | Gain vs pause |
|---|---:|---:|---:|---:|---:|
| 9901 | 2.479049 | 2.468092 | 2.483098 | -0.010957 | 0.004050 |
| 730201 | 2.456964 | 2.525914 | 2.537470 | 0.068950 | 0.080506 |
| 730237 | 2.474577 | 2.429517 | 2.490984 | -0.045059 | 0.016407 |

| Seed | Ordinary contrast, conditional 95% interval | Pause contrast, conditional 95% interval |
|---|---:|---:|
| 9901 | [-0.043969, 0.022655] | [-0.028121, 0.038447] |
| 730201 | [0.032677, 0.103579] | [0.044612, 0.116845] |
| 730237 | [-0.077033, -0.015284] | [-0.019885, 0.052422] |

The three-seed mean gain over ordinary is **0.004311 nats**, paired SD 0.058518 and nominal t interval [−0.141055, +0.149677]. Against pauses the mean is +0.033654, SD 0.041042 and interval [−0.068301, +0.135609]. Three existing development seeds provide weak variance evidence. Conditional source intervals do not remove this training-seed uncertainty.

Exact answer-prefix accuracy is near the floor: latent 1.21%, 1.81%, 0.86%; ordinary 1.73%, 0.60%, 1.12%; pauses 0.86%, 1.12%, 0.86%. These are operation-bin macro averages, with no termination scoring. First-answer NLL and every operation bin are included in the complete receipt. First-token contrasts are not consistent evidence of calculation ability.

The existing broad-loss result and its failed useful-margin replication gate remain unchanged. This new panel addresses direct source-authored calculations without supplied work; it does not demonstrate transferable arithmetic, content-sensitive reasoning, allocation or training efficiency. Latent training consumed 3.782219 times the ordinary matrix work at equal source exposure.

All 27 independent cached/full-prefix records and 27 batch/individual checks pass; maximum log-probability discrepancy is 3.20e-14. CPU readout takes 518.221 wall-seconds, peak worker RSS 1.69 GiB, zero chips or updates. All source and checkpoint hashes are retained. Both fresh confirmation reserves remain sealed; terminal alpha is zero. No training follows automatically.

The training census found only 617 matching direct calculations with intact packed premises. Limited exposure is a plausible alternative to a mechanism limit, but this diagnostic does not establish that more exposure would help. Any exposure study needs a separate frozen plan, ordinary controls, and a check that its longer training prefix does not introduce exact evaluation expressions.

[All scores, intervals and numerical checks](report.json) · [Frozen readout](../../../docs/DIRECT_ARITHMETIC_READOUT_20261030.md) · [Source and training-exposure census](../source-preparation-v1/REPORT.md).
