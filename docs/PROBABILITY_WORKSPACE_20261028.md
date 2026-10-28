# Probability-embedding workspace: numerical reference and prospective screen

The old discrete-note, sparse hidden-feedback and frozen-feature mechanisms
remain closed with their recorded negative outcomes. The owner renewed the
workspace objective and authorized continuing with a different bounded design.
This reference tests a published operation rather than another correction to
the frozen predictor's final representation.

Primary source: [PonderLM v3](https://arxiv.org/html/2505.20674v3), equations7--8.
The official implementation was inspected at commit
3deec205194d754f3e9641182e0f4c3ea3c3f6d4 of LUMIA-Group/PonderingLM.
The implementation here is independently written. This is not a reproduction
claim: backbone, data, optimizer and scale differ. Full-vocabulary expectation
matches the inspected active reference path; top-k is not silently substituted.

At each causal position, the model predicts a vocabulary distribution and
forms its probability-weighted input embedding. Add this proposal to the
current embedding stream and recompute the complete causal language model.
Differentiate the final ordinary natural-token NTP loss through every replay
and expectation. Targets never enter proposals. No external teacher, generated
source examples, reward labels or policy gradients. Zero extra passes must
exactly reproduce the ordinary model, including gradients and optimizer update.

This differs from the closed sparse experiment: it returns a distributional
proposal to input-embedding space and recomputes every layer and causal position;
it does not append a raw final hidden state to cached same-position keys/values.
It remains extra tied computation and must earn its cost. The name is not
evidence of reasoning, allocation or efficiency.

First qualification: small numerical arrays, all16 chips/64 ranks, five
disposable AdamW updates with zero/one/two replay and uniform controls, followed
by a separately launched exact full-state replay. CPU SDPA and cross entropy
are independent references for explicit Native attention and stable target loss.
Original gates:2% global and embedding/body gradients, optimizer update and
moments;0.003 loss; exact model/optimizer/RNG restore and subsequent replay;
identical replicas; no fallback beyond tensor copying. No research data or
scientifically eligible weights. Thirty-minute overall and twenty-minute phase
caps,128GiB worker,512GiB host reserve,35GiB disk reserve,10000 inodes;2GiB
artifact allowance. Preserve every failed attempt and its costs.

Prospective next stage, not admitted by unit tests: qualify actual model shape
and100 warm updates on natural TRAIN sources, then freeze a finite learning
budget from that measured execution. A roughly60--70M model is a tractable
mechanism screen from random initialization, not the full from-scratch campaign
or evidence about billion-parameter capability. Before fitting, specify supported
replay budgets and their training schedule, ordinary equal-data and equal-work
endpoints, fixed tuning opportunities, complete cooldowns and held-out sources.
The old320-group confirmation reserve stays untouched. No larger backbone or
allocation campaign follows automatically.

Controls: an independently trained ordinary predictor; a uniform expected
embedding (content-free) control; within-model zero/one/two exits only when
explicitly supported during training. The uniform control has no proposal
forward/backward and is explicitly cheaper. Count all vocabulary readouts,
expected-embedding matrix multiplications and body recomputation; report actual
time separately. Uniform-control harm or donor corruption alone cannot establish
reasoning. Require absolute useful predictions, same-checkpoint supported-depth
gain, improvement over an independently trained shallow control, and eventual
authentic dependency transfer. Broad matched-work superiority remains unproven.

Separately, a TRAIN-only audit reconstructs the exact32-of159 target selection
in the previous8192-view branch screen and counts source-authored arithmetic
chains/strict Python bindings. Exact decoded-token alignment is required. Its
coverage is a lower bound; truncated/unsupported text is not classified as
uncomputable. No model scoring, new held-out selection or gate changes.
