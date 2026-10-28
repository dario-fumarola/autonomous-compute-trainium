# A small advantage over a matched local nonlinear correction

The new local control completes all2048 updates from initialization and passes
its final gradient/full-state audit. All models below use the same frozen C0,
the same8192 fresh TRAIN views,and the same sampled natural targets.

| Predictor | Full32 development NLL | Gain over frozen C0 |
| --- | ---: | ---: |
| Frozen C0 | 2.282112 | — |
| Linear correction | 2.283227 | −0.001115 |
| Local nonlinear correction | 2.277691 | +0.004422 |
| One attention/FFN step | 2.275685 | +0.006427 |
| Two shared attention/FFN steps | 2.276055 | +0.006057 |

The local nonlinear control and the one/two-step models each train655360
parameters. The local control has no additional access to earlier feature
positions;its input is the current frozen representation,which already contains
the ordinary model's context. Its wider local FFN uses the parameter budget
that the other models spend partly on attention. Initialization predictions,
input projection,source order,target RNG and optimizer settings match;this is
an allocation control,not a direct intervention on an identical trained model.

One-step attention beats the local control by0.002005nats,conditional95%
source-group interval[0.001015,0.003098]. Two steps beat it by0.001635,
[0.000420,0.002954]. The local control's own gain0.004422 has interval
[−0.000106,0.008391]. These are one-seed,reused-panel development comparisons;
they do not capture training variance or establish equivalence/general reasoning.

The attention architecture has a small advantage over this trained local
alternative. This is consistent with a benefit from the additional context
read,but does not causally isolate its content:attention and local FFN
capacity are allocated differently,and no content intervention was performed. Ordinary nonlinear correction also improves
the point estimate substantially. Two steps still do not improve on one.
The original0.01/0.005 iterative gates remain failed;no gate or endpoint changed.
No matched-total-work ordinary continuation comparison has been won.

All four local branch parameter gradients agree with FP64;maximum relative
error2.48657e-6. Model/AdamW/RNG restore exactly,and the scientific checkpoint
is unchanged by the audit. Frozen C0 reproduces the same2.2821122052683624.
Eight focused CPU tests pass,including locality,parameter inventory,initial
identity,causal workspace checks and exact next-update replay. The fit takes
348.25seconds,the mature audit/TRAIN diagnostic42.68seconds,and the complete
controller403.31seconds. No additional Native job or new feature extraction.
Actual head/backward matrix work and all shared prior work remain in the
[cost ledger](complete-costs.json).

The combined diagnosis is narrower than a thinking claim:repetition overfits
these small branches;fresh source diversity enables a transferable correction;
the attention branch leads its local alternative;another iteration does not
improve held-out prediction.
Math NLL still worsens in the attention conditions despite web/code gains.
The next mechanism question concerns the information used by the extra context
read. It should be tested on authentic dependencies with matched local controls,
without converting this development result into a claim of reasoning or
permission for a larger pretraining campaign.

[Registered control](../../docs/RESIDUAL_LOCAL_CONTROL_20261028.md) ·
[All contrasts](report.json) · [Audit](train-diagnostic.json) ·
[Source-volume experiment](../residual_data_volume_20261028/REPORT.md)
