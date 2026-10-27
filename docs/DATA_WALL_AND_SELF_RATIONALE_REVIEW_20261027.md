# Review of the data-wall and self-rationale proposals

Status: research notes, not an adopted charter, admitted experiment, or change
to the running precision-corrected rate comparison. The owner asks for judgment
and continued work. Finish that comparison before adding another mechanism.

The useful common idea is to separate predictive improvement, resource
efficiency, allocation, and causal use of intermediate content. Each deserves
its own test. None of the proposed definitions turns the completed negative
comparisons into positives.

The existing fixed-depth loop does not learn allocation by construction.
Content-specific causal use is unestablished, rather than a demonstrated
failure: the required content intervention was not established by the old
quality comparison. Input-dependent hidden computation can also be generated
without readable token sampling. Information-gain policy gradients are a
candidate learning signal, not the only possible teacher-free formulation.

## Self-rationale proposal

Retain the content controls and supported inference-budget measurements.
They would make a future thought experiment more informative than a loss-only
comparison. However:

* Sparse fixed anchors or entropy thresholds are allocation policies, not
  evidence that the model learned when computation has value. Compare against
  random placement and fixed placement at the same measured cost. A full
  vocabulary confidence readout also costs computation.
* Shuffling or masking thoughts changes the predictor's input distribution.
  Include matched length, boundaries, positions and compute, and qualify the
  intervention. A training arm denied thoughts and an ablation of one trained
  checkpoint answer different questions. Neither a 50% point-estimate ratio
  nor a distribution-shift penalty proves mediation by useful reasoning.
* Measure a curve over lengths supported during training, with uncertainty
  and equal-resource alternatives. Improvement at useful finite budgets is
  evidence even if a forced longer thought eventually hurts. Strict
  monotonicity to 32 tokens is a design target, not a universal definition of
  thinking. Readability is also distinct from predictive utility.
* Keep future source tokens out of thought generation, selection and policy
  inputs. They may grade TRAIN thoughts. Evaluation thought selection must
  depend only on the prefix. Preserve source-group separation and the fresh
  confirmation reserve.

[RLP](https://arxiv.org/html/2510.01265v2) studies Qwen3-1.7B-Base and
Nemotron-Nano-12B-v2, starting from existing checkpoints. Its experiment samples
one rewarded target per document; its reported FLOP-matched continuation can
use 35 times as many input tokens. Its shared network receives a clipped
policy-gradient update on thought tokens. Adding a separate ordinary NTP term
would be a proposed variant, not an exact reproduction. Subtracting the same
no-thought scalar from every member of a group cancels when forming centered
group advantages; an EMA baseline does not by itself supply an additional
policy-learning signal after that centering.

[Quiet-STaR](https://arxiv.org/html/2403.09629v2) starts from Mistral 7B.
These studied scales do not establish a universal 1–2B capability floor or
prove that small-model samples cannot carry useful information.
[PonderLM-2](https://arxiv.org/html/2509.23184v4) includes from-scratch 410M
experiments and estimates about eight times vanilla training FLOPs for its
Jacobi recipe. Its two-times inference comparison is a different resource
constraint. The proposed simpler continuous arm needs its own qualification
and whole-training accounting; it does not inherit those results.

Consequently, do not assume 2–3 times overhead from sparse anchors alone.
Count sampled group size, thought lengths, reward spans, all readouts, cache
work, baseline forwards and backward passes. The measured local 956M path
processes 94,239.927 real targets/second: 100B targets is 12.281 warm days;
10–20B at a hypothetical three-times cost is 3.684–7.369 warm days per arm,
before qualification, audits and controls. This is an extrapolation, not a
new mechanism benchmark. External rental prices and throughput are unverified.
The owner has relaxed overall clock admission, so time alone is not a reason
to call a foundation impossible.

An [official OLMo-2 1B base exists](https://allenai.org/olmo2). Borrowing it would
test continued pretraining, not the current random-initialization claim.
Warming up a locally initialized model and then introducing thoughts can still
be a from-scratch experiment if its entire history and common-prefix costs
are charged. Policy gradients and self-generated thoughts remain outside the
current phase; this review does not enable them or change AGENTS.md.

## Data-wall proposal

Fixed unique source material is a legitimate new resource constraint. It must
be selected for an intended application, with a repeated-data ordinary
baseline tuned for that regime and a larger ordinary challenger reported as
a separate parameter-budget comparison.

[Muennighoff et al.](https://arxiv.org/html/2305.16264v3) find relatively small
repetition penalties through four epochs in their setting. Applying their
15.4 fitted repeat scale gives 93.15% total fresh-equivalent exposure at four
epochs; the marginal value at the endpoint is 82.30%. Those are different
quantities. Four epochs is not evidence that the ordinary model has run out
of useful learning. Transfer of that fitted law to this corpus is a hypothesis.
[The weight-decay study](https://arxiv.org/html/2509.14786v1) motivates tuning
regularization; its 0.8 result is not a universal setting for every model and
optimizer. A 301M-token tune that never repeats a 4.1B corpus cannot establish
the best regularization for four epochs.

The accompanying arithmetic has several implementation corrections:

* Four of the five proposed shapes violate the current four-KV-head GQA
  divisibility rule. Some proposed FFN dimensions are fractional as well.
  Legal illustrative replacements are enumerated in the linked calculation;
  they vary head dimension and are not admitted or exactly parameter-matched.
* A 20-to-28 block-application change is 1.4 times body work, but about
  1.345946 times whole neural matrix work at 1,024 positions under the current
  Family-A estimator. At that estimate, 16.4B ordinary targets buys 12.185B
  loop targets. Optimizer, boundaries and actual execution still need counting.
  Multiplying runtime by loop overhead again after reducing targets to match
  work double-counts that overhead under an equal-throughput approximation.
* RMSNorm of the injected input, a post-core norm and half-sized residual
  updates do not preserve the ordinary function at initialization. A separate
  ordinary bypass could preserve it; the changed recurrent graph needs its
  own qualification. A bounded diagonal coefficient does not guarantee that
  the entire nonlinear recurrent transition is contractive. Affine norm
  parameters also exceed the proposed 1,536 adapter-only count.
* Weight sharing does not imply cache sharing. With conventional separate
  FP16 K/V per logical application, the proposed cache is 28,672 rather than
  20,480 bytes/token. A cache-sharing scheme changes execution semantics and
  needs separate validation. Larger-model KV size does not generally scale
  directly with total parameters.
* An all-token vocabulary head smaller than an eight-block core is still
  larger than computing that core on only 20% of tokens: the ratio is 1.953125
  before other routing costs. This does not rule out cheap hidden-state gates.

The current qualified recurrent scale candidate has 375.94M deployed
parameters, not the old 142M inventory. Arguments based on the old 14M core
do not automatically apply to its larger body.

## Statistical and decision corrections

The rule “mean gain at least 0.01 and lower bound above zero” tests a different
claim from “lower bound above 0.01.” Neither is inherently invalid. The latter
requires evidence for a useful gain, rather than just a positive gain. Do not
replace it after seeing results or call it a mistaken significance test.

At a true gain of exactly 0.01, an unbiased symmetric continuous estimate has
only 50% probability of meeting the mean-at-least-0.01 clause. Thus the proposed
combined rule cannot have 90% power there, regardless of seed count. Power
must specify an alternative above the actual decision boundary.

Two or three paired seeds estimate variance poorly. Using the review's rounded
Family-B differences gives SD 0.000774, but its normal-theory one-sided 95%
upper bound for the underlying SD is 0.012348. Pairing is useful; it does not
make this tiny variance estimate reliable. Keep the existing prospective
fresh-source and at-least-five-fresh-seed confirmation requirement, with power
and repeated-testing allocation declared before confirmation.

A hard-slice win can be a mechanism finding. It cannot replace a broad-loss
claim retroactively. High reference loss is not proof of an unknowable token,
and the failed prefix probes do not prove all future selectors impossible.
The target-informed oracle measures hindsight selection opportunity, not
0.078 nats of identified reasoning or a recoverable causal-router ceiling
across different models. Mean-loss noninferiority also needs an uncertainty
bound if used as a decision criterion.

The proposed “stop at k=4 failure” rule would prevent observing a later
“only k=8 wins” outcome. Decide which question is intended before launch.
No new data-wall, RL, borrowed-base, routing or confirmation campaign is
admitted by these notes.

Reproduction: `python -m scripts.audit_oct27_review_arithmetic` produces
[the small arithmetic receipt](../results/proposal_review_20261027/arithmetic.json).
It accesses no corpus, weights or held-out examples and uses zero chips.
