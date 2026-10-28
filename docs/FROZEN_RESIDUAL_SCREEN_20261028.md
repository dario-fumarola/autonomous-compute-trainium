# Can a small workspace learn to improve a fixed predictor?

The previous one-step whole-backbone feedback architecture is closed. Its
inference-only scale/cache interventions did not rescue it. This new bounded
screen isolates learnability: freeze C0 and its vocabulary head, cache causal
final representations, and train only a small residual branch in that space.
No base prediction can deteriorate during fitting because no base weight moves.
The residual output projection starts at exactly zero. This is a pretrained
mechanism diagnostic, not the original random-initialization claim.

The workspace projects frozen representations from2048 to128 dimensions.
Its shared cell attends only to prefix representations at or before each
query, then applies a residual feedforward update. Queries are independent.
The branch output is projected back to2048 and added to the original final
representation before the unchanged vocabulary head. No generated labels,
teacher,PG,learned halting or changes to the ordinary model are introduced.

Three conditions receive identical source views and sampled target positions:
linear low-rank residual correction;one cell execution;two shared cell
executions. The iterative conditions have exactly the same parameter count
and initialization. The cheaper linear control has fewer active parameters.
No unused auxiliary head or inactive cell is retained in the linear model.
This tests whether iteration contributes beyond cheap extra predictor capacity.
It does not alone establish reasoning or inference-time scaling:the two-pass
model's shallow exit is not trained and will not be reported as a valid exit.

Freeze the first2048 original TRAIN views and the same448 reused TUNE views
already used in the workspace program;verify source-group separation. Keep
the60/20/20 web/math/code evaluation weights. All inputs and targets are
authentic source tokens. Each view has159 possible targets;sample32 distinct
positions uniformly without replacement,using an explicit shared RNG. Four
views per update,512 updates,one source traversal,65536 sampled natural labels
per arm. This is an unbiased sparse uniform-NTP estimator,not a selected-target
bonus. The frozen base supplies representation features,not new targets.

Use seed7285,AdamW LR3e-4,weight decay0.1,gradient clipping1.0,no LR search,
early stopping or endpoint selection. Save the full small branch/optimizer/RNG
state and actual telemetry. Exact zero-output identity,causality,query
independence and repeated-cell finite-difference gradients must pass first.
An independent HF full-size CPU reference must qualify the160-position Native
feature extractor before source caching. The base stays unchanged. Store FP32
features so quantization is not silently added to the intervention.

The all16-chip extraction is bounded to10minutes and192GiB worker memory.
The complete registration lasts45minutes. CPU fitting uses up to three16-thread
workers,with a30-minute cap each. The first16 actual fitted updates measure
whether the remaining fixed fit/evaluation fits the cap;they are retained and
charged,not qualification-free training. Resource guards retain512GiB host
availability,35GiB filesystem space and the existing inode reserve. No native
training port is needed for this small branch screen. A failed budget gate
stops the fixed experiment;it does not renew it or change the endpoint.

Primary development readout is full32 continuation NLL,from the exact same
frozen C0 baseline. Its weighted raw score must reproduce2.282111260797275
within1e-5 and agree across all arms. Source-group bootstrap intervals are
conditional on one fitted seed and reused development sources. A promising
iterative mechanism requires at least0.01nats improvement over frozen C0,
positive lower conditional interval,and at least0.005nats improvement over
both linear and single-step controls with positive lower intervals. These are
screening gates,not confirmation. A linear-only gain is extra readout capacity;
a two-pass gain without beating the controls is not an iteration result.

Report every arm,source exposure,branch parameters,full vocabulary-projection
cost,CPU/Native time and all failed work. The comparison is equal data,not
matched total training compute. Frozen-base feature reuse is an engineering
amortization,not permission to erase foundation,prefix,readout or backward
costs. An improvement must later survive a properly tuned full ordinary
continuation and equal-resource controls. No large fit,router,confirmation or
from-scratch campaign is admitted by this screen.
