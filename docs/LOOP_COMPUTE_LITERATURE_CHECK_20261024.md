# What recent loop results do and do not establish

Primary-source check on October24, while the frozen variable-depth comparison
is still running. This note changes no live recipe, endpoint, exit or analysis.

## Truncation can improve loss without improving the loop mechanism

[Schwethelm et al., sections4–5 and appendixG](https://arxiv.org/html/2604.21106)
study116 iso-depth pretraining runs on FineWeb-Edu with a32k tokenizer. Their
fitted recurrence-equivalence exponent is0.46, rather than full equivalence
to independent layers. Truncating backpropagation lowers validation loss at
fixed compute, but reduces the fitted exponent from about0.45 to0.38; the
saved work buys more data. Two-lane hyperconnections raise the exponent to
about0.65, with two-pass models competitive at some budgets. The authors also
identify a missing analogous hyperconnection intervention in the ordinary
reference as a limitation. Their Muon-based optimizer, learned injection,
width sweep and packed2048-token evaluation differ from our experiment.

Our inference: do not accept truncated gradients as an automatic cure for
sharing, or transfer their exponent numerically to our architecture. If a
future change improves residual information flow, an ordinary model with the
same applicable change is a necessary control. A better loop-versus-loop
result alone cannot establish our efficiency objective.

## Fixed-point benefits are a different claim from ordinary-model efficiency

[Huang et al., sections3–5, tables1–3](https://arxiv.org/html/2610.06833)
train100M,400M and1.6B models on21.5B,85.9B and343.6B tokens. Their learned
depth prior and orthogonal input injection improve looped-model perplexity
and support terminal KV sharing. The parameter-matched four-layer ordinary
reference uses substantially less training work. The twelve-layer ordinary
reference has more parameters and lower perplexity at every reported scale.
These are informative controls, not a demonstrated simultaneous match on
parameters, training compute and prediction quality. The reported training
mixture is multilingual and includes TxT360; we have not independently
audited every source against our natural-source-only policy.

Our inference: depth variation, state convergence and cheap cache reuse are
separate measurements. Our all-token parallel NLL does not test their terminal
KV caching claim. Do not silently import learned routing, truncation or data
from that recipe into the current study.

## A generation win is not necessarily a perplexity win

[OperLoop, section5.2 and appendixC](https://arxiv.org/html/2609.37379)
reports matched-training-FLOP improvements in average generative task scores.
Its main comparisons are5B/10B MoE references and looped models with roughly
half their parameters, trained for approximately281B/503B nominal positions.
Both WikiText and LAMBADA perplexity remain worse than the corresponding
non-looped reference in the main table. This is stated in the paper itself.
It supports studying transition design, while supplying no direct evidence
that copying its update will improve our150M dense model's natural-text NLL.

Our decision remains to finish the current fixed-versus-variable comparison,
independent audits and sealed analysis. These papers motivate bounded future
tests; none explains away our completed negative compute comparison or
justifies scaling the present recipe on faith.

## Primary-source recheck, October26

This read-only recheck changes no active recipe or budget.

[RINS, abstract and section3](https://arxiv.org/html/2502.07503v3) explicitly
studies fixed model size and training compute, reducing recurrent models' token
exposure accordingly. Its positive result contradicts a universal prohibition
on dense recurrence winning under that constraint. It does not establish a
break-even exposure or expected effect for our topology.

[Parcae, appendixO](https://arxiv.org/html/2604.12946v1) includes140M/370M
fixed-parameter, fixed-FLOP experiments with token budgets adjusted for depth.
It separately trains depth-one Parcae references. That supports a within-family
comparison, while leaving the strongest conventional backbone a separate
question. Our prepared eight-layer control addresses that question; it has
no learning result yet. Parameter and work-accounting conventions also differ.

[Truncate or Not, sections5.2 and6](https://arxiv.org/html/2610.26030v2) finds
that fewer fully differentiated loops beat deeper truncated alternatives when
forward depth is also selectable. Its fixed-size and freely sized model
frontiers answer different resource questions. This supports retaining our
small fixed-two/full-gradient candidate while finishing strong controls. It
does not predict that our current gain will survive those controls.

Decision: complete the frozen development comparison and qualify the eligible
stronger baselines. None of these papers justifies an automatic30B run, a
portable Chinchilla-constant forecast, or a claim that the scientific ranking
is explained by the newly localized mixing precision mismatch.
