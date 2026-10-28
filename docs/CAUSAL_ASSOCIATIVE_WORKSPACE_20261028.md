# A concrete workspace operation: learning which past outcomes apply now

This is a new causal-memory screen,not another fit of the closed two-step
residual mechanism. It tests a concrete operation:associate an earlier prefix
representation with the token that actually followed it,and learn when that
association helps the current prediction. At query t,only pairs(h_j,x_{j+1})
with j<t are available. x_{t+1} is a scoring target only. The original prefix
already contains every stored outcome;there is no future-conditioned writer,
external teacher,synthetic source,or new information beyond that prefix.

This follows established neural-cache/fast-weight work,not a novelty claim:
[Grave et al.](https://arxiv.org/abs/1612.04426) retrieve past hidden-state/token
pairs;[Clark et al.](https://aclanthology.org/2022.emnlp-main.661/) learn to use
past prediction gradients. The present screen is a learned cache reader,
not a reproduction of the latter gradient-update algorithm. Useful copying or
contextual memory is not by itself general reasoning or invented thought.

Keep C0,its head and all cached features frozen. Each query mixes the ordinary
distribution with a normalized distribution over previously observed tokens.
The cache distribution sums attention weights for matching token values.
Gate inputs are the current prefix representation only;neither current target
identity nor whether that target appears in memory may enter the gate.
An exactly zero projected mixture strength initially gives ordinary prediction.
The strength is projected into[0,1] after AdamW;lambda=.5*strength*sigmoid(gate).
This learns whether to USE memory after its read cost is paid,not autonomous
allocation of computation or a claim about skipping work.

Three fresh seed8317 conditions share source/query schedules and a gate:
- Bag:uniform weight on all earlier pairs;only token frequencies and the gate.
- Cosine:full2048-dimensional frozen-prefix cosine similarities;learn a
  temperature and the same gate,but no addressing projection.
- Learned:independent2048-to128 query/key projections,initially identical;
  learn addressing,temperature and the same gate.

Temperature starts16 and is projected into[1,256]. All gates start with zero
weights/bias. Use AdamW LR3e-4,WD0.1,clip1,batch4,1536 updates and32 uniformly
sampled natural targets per view;no auxiliary target weighting or LR search.
Projected updates and their optimizer moments must restore and replay exactly.
Count different parameter/work inventories;this is not parameter matching.

Use only the existing8192 TRAIN feature views. Entire source groups whose
first32 hash bits are0 modulo8 form calibration:797 views/482 groups. The
first6144 remaining views in original order form fitting data;calibration
groups are entirely excluded. Freeze target sampling with RNG8318 separately
from model initialization. Calibration uses the final32 natural targets.
All448 original TUNE views are evaluated only after fitting and mature audits.
The320 reserved groups,TEST and old confirmation pool stay untouched.

A strong nontrained cache is also selected exclusively on calibration groups:
temperature{0,4,16,64,256} by mixture weight{0,.005,.01,.03,.1,.3,.5},using
the fixed60/20/20 mean. Zero weight is raw C0. Report boundary choices and
the whole calibration grid. No per-domain endpoint or favorable readout is
selected. Learned models have one fixed endpoint each;no recalibration.

Report raw C0,calibrated fixed cache,all three learned gates,previous one-step
residual control,and domain means. For the learned reader,rotate memory VALUES
by one within each allowed prefix while preserving keys and the prefix's token
counts. No future value enters that intervention;recompute the cache read.
The bag model is invariant to this rotation. The intervention may still cause
distribution shift and does not establish reasoning alone.

Primary advancement:learned addressing improves raw mean NLL by>=0.01 and
beats EACH of calibrated fixed cache,trained cosine gate,and trained bag gate
by>=0.005,with positive conditional source-group lower95% bounds. Also require
aligned key/value pairs to beat the rotation by>=0.005 with a positive lower
bound. Report repeat versus nonrepeat targets as a prespecified descriptive
split;the current target defines that analysis label only,never a policy input.
If only a standard cache wins,report exactly that positive control and do not
call it a learned-thinking result. A repeated-development-panel result is not
confirmation;even a pass needs fresh sources/seeds and a properly costed
ordinary continuation with the same cache opportunity.

Before fitting,test target probability against an explicit full distribution,
current/future-target and future-feature invariance,empty memory identity,
value-rotation causality,gradient flow,projected-boundary learning,and all named
gradients against a sequential truncated-prefix reference. At actual shape,
compare FP32 with FP64 on one fixed TRAIN example/domain at nonzero strength,
then again at each trained endpoint;retain2% gradient and.0002 loss gates.
Check full model/AdamW/counter restore plus one disposable update replay.
Targets enter only likelihood scoring;frozen base target log probabilities may
be cached to avoid repeating an identical head calculation during fitting.
Those forwards are charged,as are inherited C0/features and all model reads.

This is a CPU mechanism screen with45-minute registration,8-minute raw-cache
preparation,10minutes per fit and the remainder for audits/evaluation. Three
16-thread fits may overlap while resource floors hold. Measure the first16
updates before retaining the rest. Keep512GiB host availability,35GiB output
space,10000 inodes and48GiB per-process RSS. No new Native/backbone update is
needed. Preserve every failed fit and full state. No larger run,automatic
router or confirmation follows from a favorable exploratory number.
