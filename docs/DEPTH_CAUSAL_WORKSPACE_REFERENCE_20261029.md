# Depth-causal workspace: prospective reference and feasibility only

The completed learning evidence must remain unchanged. The original fixed-one
replication misses its useful-margin gate. Preliminary supported-budget scoring
also fails to show a second-step advantage; its exact readout is still pending.
Neither finding admits the prepared ephemeral fit or a larger training run.

One unresolved design issue is that the persistent graph changes all earlier
source representations when the thought budget changes. Its zero/one/two modes
share weights, but their earlier causal memories differ. This is a legitimate
whole-model budget comparison; it does not isolate one additional operation
on the same lower-stage representations.

The next CPU reference will instead allow a query at source position t and
stage k to attend only positions s <= t at stages j <= k. Stage zero contains
ordinary source embeddings; stage k+1 takes the previous stage's final hidden
state as its input. All stages share the same body and vocabulary head. Source
and latent states use their original source rotary positions. Earlier stages
cannot depend on later stages. This follows the parallel-mask construction in
[Pretraining with Token-Level Adaptive Latent Chain-of-Thought](https://arxiv.org/html/2602.08220v2),
section3.2. The first reference omits that paper's router, pruning, hidden-state
mixture and adaptive loss. It is not a reproduction or a claim of novelty.

Use the existing51,747,328-parameter six-block model, with budgets0/1/2 and
ordinary uniform natural-token NTP. No additional learned parameters, budget
embeddings, new labels, changed source mixture or target emphasis. Keep every
lower-stage key/value tensor differentiable while reusing it. There is no
numerical fixed-point solver: the graph is exact after one body execution per
stage. Retain a slower full-recomputation implementation as an independent
reference. A cached implementation must agree in logits, generated states and
all parameter gradients, including memory paths.

Required software checks before any Native execution:

- Budget zero agrees with the ordinary path in outputs, full gradients and an
  optimizer step from identical state.
- Full-mask and cached implementations agree at all three budgets, in FP64 and
  FP32, including full gradients. Parameter names and initial values match.
- Extending the maximum budget leaves every earlier stage unchanged.
- Changing future source tokens cannot affect earlier outputs or latent states.
- Token-by-token exact inference agrees with parallel training; both must use
  the same two-dimensional causal condition, not an ordinary triangular mask.
- A deliberately detached cache fails a nonzero gradient-path check.
- The ledger includes all stage projections, attention rectangles, final head,
  backward work and actual recomputation; reuse is counted only if executed.

Numerical fixtures establish software behavior only. No held-out examples,
scientific updates, reserve access or learning claim are admitted by this
reference document. Any actual-shape qualification requires its own immutable
plan, all16chips, cold/mature CPU gradient comparison,100warm updates, exact
optimizer/RNG restoration and measured resource/cost guards. Any later fit
requires a separate finite protocol and unchanged strong ordinary controls.

A stable lower-stage graph is an architectural property, not guaranteed
retention after shared-weight training. The zero-budget quality guard remains
necessary. Likewise, removing solver work could lower execution cost without
improving predictions. A useful result must establish both separately.
