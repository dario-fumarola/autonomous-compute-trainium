# Can observed prediction errors supply useful internal work?

The learned cache is closed after its two failed screens. This is a distinct,
zero-update diagnostic: a temporary vocabulary-bias update from past prediction
errors, rather than a positive probability mixture over past token identities.
It is an established dynamic-evaluation idea, not a newly invented reasoning
algorithm. See [Krause et al.](https://arxiv.org/abs/1709.07432) and
[Clark et al.](https://aclanthology.org/2022.emnlp-main.661/). The present fixed
bias update does not reproduce their learned fast-weight architecture.

At position t, the frozen base supplies probabilities p_j and representations
h_j. Only j<t may write e_j = onehot(x_{j+1}) - p_j into temporary memory.
The workspace is b_t = sum_j a_tj e_j. Predict with softmax(logits_t + eta*b_t).
This is exactly one negative-gradient step on a weighted prefix-only NTP
objective with respect to a fresh zero vocabulary bias. The bias resets for
each query; no persistent parameter or checkpoint is updated. Attention weights
are frozen cosine similarity softmax at temperature16. Current/future labels
never enter weights or writes. Earlier teacher-forced outcomes are observed
prefix information, not answer-aware workspace selection.

This tests an operation before learning its use. It cannot establish that a
model invented an algorithm, learns to think from random initialization,
allocates compute, or beats ordinary training at matched work. A future learned
version needs its own protocol and a strong ordinary model with the same
adaptation opportunity. The current source-group reserve remains protected.

Use the existing797 TRAIN calibration views/482 groups, already frozen and
disjoint from prior branch-fitting groups. This is reused calibration, not
fresh statistical confirmation. Compare four separately calibrated methods:
cosine error, uniform error, cosine label-only (sum a*onehot), and cosine
prediction-only (-sum a*p). The last two distinguish actual error correction
from token reinforcement or probability flattening. Give each the same eta
grid {0,0.25,1,4,16}; zero is unchanged C0. Select on the60/20/20 mean across
the final32 natural targets; report all grid and domain scores and boundaries.
Do not select domains, target positions, horizons or temperatures by outcomes.

CALIBRATION gate: the cosine error method must improve raw C0 by>=0.005 and
each of the independently selected label-only and prediction-only methods by
>=0.002. If it fails, publish the complete calibration result and stop without
another TUNE read or fit. This is a necessary screening condition, not a bound
on all possible future fast-weight learners.

Only if calibration passes, read all448 existing TUNE views exactly once with
the four frozen choices, raw C0 and cosine-error values rotated within each
allowed prefix. Preserve paired probability/outcome error vectors in rotations.
Require error gain>=0.01 over C0,>=0.005 over each label/prediction control,
and a positive association contrast, all with positive conditional source-group
lower95% bounds. Report uniform-error control, all domains and prior one-step
adapter scores too. A lead is exploratory and needs fresh sources/seeds; no
automatic fit or confirmation. All320 reserved groups,TEST and the original
confirmation pool remain untouched.

Numerics: test strict future-feature/logit/label invariance, empty-memory and
eta-zero identity, normalized distributions, saturated correct-class errors,
uniform-rotation invariance, and agreement with an independent literal weighted
cross-entropy bias gradient. At actual shape, compare FP32 analytical errors
with FP64 autograd on one TRAIN view per domain and retain the2% relative-error
gate. Reproduce cached calibration raw log probabilities and prior TUNE C0
scores within0.0002. Hash all used code, features, source maps and checkpoints;
verify unchanged head versions and hashes after scoring. Preserve failures.

CPU only,16 threads,20-minute registration and15-minute scoring cap. Project
duration after the first16 calibration views. Retain512GiB host available,
35GiB free output space,10000 inodes,48GiB maximum process RSS. No new feature
files or model states are required; store small private score summaries only.
Count all head/correction matrix operations and measured wall time, including
calibration and audits. Prior foundation,feature,failed-history and comparator
costs stay charged. This is an unoptimized headroom test,not a serving benchmark.
