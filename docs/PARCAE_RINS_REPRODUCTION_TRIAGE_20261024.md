# Primary-source reproduction check

Checked October24 while depth-connection-v1 trains. This note changes no active
model, precision, source order, endpoints or tests. The ordinary5.1B and cheap
connection pilots are already complete; their results remain preserved.

[Parcae](https://arxiv.org/html/2604.12946v1), sections5.2 and appendicesO/P/Q,
is a relevant reproduction candidate:140M/370M fixed-parameter training-budget
sweeps trade tokens against recurrence and report improved validation frontiers.
The frontier control is depth-one Parcae, distinct from its ordinary-transformer
comparison. Small-model downstream scores are mixed at lower budgets. Its
140M setup uses a2/2/2-block prelude/core/coda, QK normalization, squared-ReLU
MLPs, gated value embeddings, constrained injection, Muon/AdamW and per-sequence
Poisson depth with truncated backward passes. This is substantially different
from our20-unique-block, two-pass AdamW model. A successful reproduction must
retain both its within-family depth control and a strong ordinary comparator;
our inexpensive connection is an additional useful control. Copying only the
injection equation would be an adapted ablation, not a reproduction.

[RINS](https://arxiv.org/html/2502.07503v3), sections2–3 and appendixE, directly
compares recurrence with more ordinary training at fixed parameters and training
compute. Its repeated early half differs from our large middle core. C4 and
SlimPajama are relevant natural sources; the baseline budget is about200B tokens,
with the600M longer comparison reaching500B-token-equivalent compute. These are
far beyond our current exposure. The paper supports testing topology and budget;
it does not predict a crossover if our current recipe is simply extended.

[Iso-depth scaling](https://arxiv.org/html/2604.21106v3), sections4–5, provides
a complementary negative frontier under a width/parameter-allocation sweep.
Its sharing-efficiency estimate and truncation ablation are regime-specific.
That broader comparison and Parcae's fixed-parameter allocation experiment need
not agree about the best use of unrestricted resources.

## Checked release-code details

Pin the pre-BeNice reference at
[69284c1](https://github.com/sandyresearch/parcae/tree/69284c13746e849104f738d6d1a347b1f457df76),
rather than silently following October23's new KV-sharing code. MIT license
and Git-object provenance are retained. A CPU audit ran the pinned sampler
and tied-head cost fixture; source hashes match the connector-retrieved objects.

Three details require explicit choices before reproducing the compute frontier:

- The released sweep includes recurrence1 without changing the Poisson sampler.
  Clamping zero draws to one gives expected depth1.367879 at requested mean1;
  it is not deterministic one-pass execution. The historical paper runs may have
  used other overrides. Resolve this ambiguity rather than equating the two.
- Per-sequence masking executes to microbatch maxima. At requested mean8, a
  32-sequence software fixture averages14.36 dense forward calls, versus7.98
  logical passes per sequence. Count executed work and padding separately.
- The released cost estimator drops the tied vocabulary table and hence its
  output projection in an isolated fixture. Value-embedding gathers complicate
  the full-model estimate; this fixture does not establish the net accounting
  error or overturn the paper's measured results. Use an operator ledger here.

[Exact CPU readiness audit](../results/parcae_reference_readiness_20261024/report.json).

The small candidate component matches the pinned injection's outputs and all
gradients; a separate finite-difference check passes. Truncation preserves the
forward result and matches an analytic tail-gradient fixture. These are software
checks, not language-learning results or a complete reproduced model. Prelude
normalization, C projection, value embeddings, initialization, optimizer groups,
sampling, data and cost accounting remain full-model reproduction requirements.
In particular our existing AdamW factory does not honor the candidate's
`_no_weight_decay` annotations; it must not be reused unchanged for this recipe.

## Decision

Finish the registered common-FP32 comparison first. It isolates the additional
core execution beyond a trained cheap connection and includes both strong
longer-trained controls. Prepare a faithful small Parcae baseline next, with its
ordinary/depth-one controls and qualified precision, before a new topology search.
Do not add routing, pass embeddings or a new loss on the strength of this review.
No new hardware launch, fresh-panel scoring or terminal attempt is registered by
this note. The public candidate is not imported by the active trainer.
