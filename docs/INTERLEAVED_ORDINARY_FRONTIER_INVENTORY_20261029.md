# Prospective ordinary controls at the persistent-state parameter ceiling

This inventory precedes the first tuning-quality scores. It admits no fit and
does not replace the active, identically shaped ordinary control. A practical
training-efficiency claim must survive conventional alternatives, as the prior
ordinary8 result demonstrated in a different model family.

All rows use the same32k tokenizer, independent input/output tables, full
64-dimensional attention heads, RMSNorm, SwiGLU and512 source positions.
The input scale must remain sqrt(2.5*width), as in the current candidate. The
formula counts both vocabulary tables in parameters and the prediction head
in work. Ordinary forward work is dense dispatched matrix work per source
window, with FMA=2 and full-square attention; it is not elapsed accelerator time.

| Model | Layers | Width | FFN width | Parameters | Forward matrix work/window |
|---|---:|---:|---:|---:|---:|
| Current ordinary/candidate body |6|512|1376|51,747,328|39,426,457,600|
| Wider ordinary candidate |4|576|1384|51,743,808|36,521,902,080|
| Deeper ordinary candidate |10|448|1112|51,654,848|42,902,487,040|

The two alternatives bracket width/depth allocation without exceeding the
deployed parameter ceiling. Neither is assumed superior or qualified. They
require their own actual-shape gradient/state/replay and runtime checks, then
the same bounded LR opportunities on the same TRAIN prefix and development
panel. Select and report both alternatives without choosing extra shapes after
seeing favorable or unfavorable scores. A new finite protocol must fix budgets
before any launch; no conventional design-space superiority claim follows
from testing these two shapes.

For equal-work scientific endpoints, spend each model's measured nominal-work
budget on its own allowed number of authentic targets, preserving source order
and independent complete cooldowns. Report unequal source exposures explicitly.
Use an additional equal-data endpoint only when needed to interpret a measured
effect; do not conflate it with the resource-matched comparison. Hardware time
must be measured, and parameter savings do not imply compute savings.

This comparison is downstream of evidence that persistent generated states help
beyond the trained pause model and remain consistent with sequential generation.
It is not a reason to launch several conventional fits while the mechanism's
learning and numerical interpretation remain unresolved.
