# Assertion syntax is present; the earlier arithmetic verifier is narrow

The same10000 accepted TRAIN code documents contain10522 assertion statements,
including1859 single equalities with an integer literal on the right. There
are9003 `assertEqual` calls,2429 with an integer operand. These are syntactic
counts, not verified tasks or evidence of model capability.

Of the1859 integer-result assertions,1038 have a call in the left expression,
and1857 contain a name. Only30 have at least two explicit arithmetic nodes on
the left; computations through definitions are not counted by that last rule.
There are454 string constants containing a doctest prompt. These categories
overlap and must not be summed into an example count.

The earlier local-arithmetic verifier returned zero qualifying cases. This
census shows that assertion syntax is available, while that verifier requires
much stronger locally resolvable semantics. It does not certify any assertion's
answer, context sufficiency, source independence or transfer difficulty.
No rule has been relaxed on the basis of model scores: no model is scored.

All9997 exact tokenizer roundtrips and8500 bounded parseable files match the
previous audit. Three software-fixture census checks and33 existing parser
checks pass. The census takes41.67CPU seconds,zero chips,zero updates andzero
held-out reads. Source bytes, identifiers and target spans are not published.

[Registered census](plan.json) · [Complete aggregate counts](report.json) ·
[Earlier zero-coverage result](../train-audit-v1/REPORT.md).
