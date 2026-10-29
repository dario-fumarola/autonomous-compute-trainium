# Additional conventional control: allocate table savings to unique depth

This is an additional prospective control, declared after the original latent
development lead but before ANY conventional-frontier fits or quality scores.
It does not replace the two previously inventoried untied shapes or change an
existing plan. All three must remain visible in any subsequent frontier screen.
No scientific fit is admitted by this document.

The51.75M candidate allocates32.768M parameters to two independent vocabulary
tables and approximately18.98M to its body. Requiring conventional alternatives
to retain both tables unnecessarily restricts their capacity allocation.
An ordinary model can tie its input/output weights and spend the saved table
on unique blocks. This is a strong resource control, not an isolation of tying
alone or proof that tying is always better.

Hold width512, attention/KV heads8, FFN1376,32k vocabulary and context512 fixed.
Use the largest integer number of ordinary blocks under the candidate's
51,747,328-parameter ceiling: **11 blocks,51,179,008 parameters**. Twelve blocks
would require54,342,144 and are excluded. This derives one shape from the
resource cap; there is no post-score layer search. Its full-square ordinary
forward work is58,300,825,600 FLOPs per512-position window, including the shared
vocabulary projection. Weight tying does not eliminate head computation.

Retain source embedding scale sqrt(2.5*width), ordinary causal NTP, the same
accepted source order and independently tuned LR opportunity. Its initialization
uses the same documented initializer, but different layer count changes output
residual scaling; it is a competing model, not a parameterwise paired ablation.
Count shared parameters once and ensure embedding/output contributions both
reach the same optimizer state. No unused independent head remains. The common
optimizer rule exempts the embedding table AND the independent output table
(the latter via additional_no_decay_parameters). The tied table retains that
exemption. This is a competing allocation/recipe, not an isolated weight-tying
causal ablation. A prior wording in the qualification-time document incorrectly
said the independent output table received decay; code and executed optimizer
groups were unchanged and both tables were exempt throughout.

Before training, require CPU forward/full-gradient/optimizer parity with an
independent tied ordinary path, followed by actual-shape64-rank qualification,
100 warm updates, independent mature gradients and exact restart. Retain all
existing resource, fallback and numerical guards. Admission is downstream of
the same completed mixed prediction/consistency gate as the other challengers.
Qualification, tuning and equal-work endpoints require new immutable measured
plans; no automatic extension or confirmation.

The disposable qualification plan is bounded to90minutes,30-minute Native
phases and20-minute CPU audit, with4GiB artifacts and both snapshots projected.
Use the existing192GiB worker cap,512GiB host reserve,35GiB output disk floor
and10,000 inode reserve. Fresh disposable seed9841 andLR0.0003; no qualification
weights are retained for scientific initialization. Scientific LR tuning would
use the same128M-target budget and{0.0003,0.001,0.003} opportunities. A defeat by
this challenger would already refute a useful advantage over the conventional
frontier; beating it alone would not complete the other two shape comparisons.
