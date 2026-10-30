# More learning improves code tracking, without a clear latent-specific advantage

All282 original targets from141 source groups are scored on the three audited
billion-target checkpoints. The original bracket-type gates fail against both
controls. Conditional source intervals do not establish equivalence, nor do
they account for training-seed variation.

| Predictor | Long type NLL | Long type accuracy | Short type NLL |
|---|---:|---:|---:|
| Generated persistent state |0.259965|89.29%|0.017038|
| Ordinary |0.281935|87.57%|0.010781|
| Trained constant pauses |0.275830|88.23%|0.015291|

The generated model's long-type advantage is0.021970nats versus ordinary,
conditional95% interval[-0.074036,0.118942], and0.015865 versus pauses,
[-0.072823,0.096461]. Neither reaches the original0.03 point threshold or a
positive lower bound. Both long-minus-short interactions span zero.

All models use the distant context: removing it raises long type NLL by0.403553
for generated states,0.314429 for ordinary and0.354716 for pauses. The generated
model's additional-context benefit over either control also has an interval
spanning zero. Type accuracy is conditional three-class prediction, not free
code execution. Short cases are near ceiling.

Unmodified long-target NLL favors generated states over ordinary by0.161705,
interval[0.005682,0.308287], but its0.030305 gain over pauses spans zero. Raw
punctuation prediction cannot replace the failed bracket-identity criterion.
The128M diagnostic remains a separate negative; these checkpoints have another
seed and greater exposure, so their before/after change is descriptive.

Nine actual-weight numerical checks pass. Independent reaggregation of all1,692
rows agrees within1.12e-16; every recovery-bound source matches at closure.
Recovery takes768.275CPU wall seconds, with zero chips or optimizer updates,
inside the original deadline. The422.503-second source-isolation stop and all
partial files remain recorded. CPU work overlaps the ordinary Native control;
cross-launch timing comparisons must retain that host-contention limitation.
No fit or fresh-reserve access follows from this result.
