# CPU FP16-body feasibility

**Both models pass the registered CPU gradient screen.** This is not native
qualification, throughput, learning or permission to change a running recipe.

At the existing numerical-only qualification weights, eight natural TRAIN
windows contribute8,180 targets per arm. FP16 autocast covers body stages;
the final norm, head, CE, masters and gradients remain FP32. Manual scale8192.
The reference is the FP32 candidate already checked against independent pinned
upstream code. Recurrence uses four forward passes and a two-pass backward tail.

| Gradient | Ordinary relativeL2 | Recurrent relativeL2 |
| --- | ---: | ---: |
| Complete objective | 0.00070893 | 0.00068231 |
| Transformer body | 0.00070944 | 0.00069775 |
| Tied input/output | 0.00070126 | 0.00066801 |
| Value embeddings | 0.00065516 | 0.00062231 |
| Injection/readout | — | 0.00056183 |

All groups clear2%. NLL changes by−0.00003438/+0.00000812, within0.003.
These are chance-level training fixture losses, not held-out performance.
The screen takes143.435s from launch through report, zero chips/updates, and
overlaps the registered scientific training. Three CPU function/stage tests
check the disabled policy, FP32 head/master/gradient invariants and AOT-eager.

[Exact errors](report.json) · [Frozen screen plan](plan.json) ·
[CPU cost](execution-costs.json) ·
[Separate native protocol](../../docs/PARCAE_FP16_QUALIFICATION_20261024.md).
