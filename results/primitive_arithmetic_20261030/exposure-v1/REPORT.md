# Explicit elementary outcomes are sparse, but unfamiliarity is insufficient

The unchanged TRAIN source census finds **9,548** aligned, intact single-
operation calculations by the 1B endpoint, representing **3,369 distinct
expressions**. Their 17,643 answer tokens are **0.001764%** of the authentic
training targets. These are counts under a deliberately narrow verified
grammar, not a census of all mathematical knowledge or reasoning opportunities.

| Training exposure | Intact elementary calculations | Distinct expressions | Answer tokens | Primitive-panel expressions observed |
|---|---:|---:|---:|---:|
| 128M | 1,275 | 801 | 2,342 | 98/256 |
| 1B | 9,548 | 3,369 | 17,643 | 163/256 |
| 3.783B ordinary endpoint | 36,831 | 8,635 | 68,387 | 201/256 |

At 1B the operator counts are 3,858 add, 1,685 subtract, 2,608 multiply and
1,397 divide. The previously completed multi-operation census separately
counts 4,398 intact calculations and 12,209 answer targets at this endpoint.
All figures retain actual packed-window boundaries and original authored
answers. The verifier supplies no target or intermediate trace.

Applying the source-only expression membership to the already completed
primitive scores gives **10/163** correct for both generated-state and ordinary
on observed expressions, and **3/93** on expressions not matched by this grammar.
Pauses score 4/163 and 3/93. This is a descriptive post-readout breakdown, with
no case selection or new gate. Missing matches do not exclude equivalent facts
in other notation. Source groups remain distinct in both strata.

The evidence therefore does not support explaining the low accuracy entirely
by unfamiliar expressions. It also does not identify a representation defect:
the 32,000-entry tokenizer already has ten single-digit numeric entries and
zero non-byte-fallback entries containing multiple ASCII digits. No tokenizer
or numerical-format change is made.

Increasing the frequency of intact computational examples in authentic text is
an exposure hypothesis worth separating from architectural changes. These
counts do not establish that such a change would work, that the existing
pretraining is insufficient in an absolute sense, or that a latent model would
benefit more than an ordinary one. No new training is admitted here, and the
ongoing frozen matched-work control remains the next scientific endpoint.

The audit reuses all 565,715 mathematical documents in the eight already bound
TRAIN shards; it scans no new source pool. Four CPU workers complete in 233.019
wall seconds, using 898.279 worker CPU seconds in total and at most 4.228 GB RSS.
Seventeen parser/source-audit tests pass. No new model scores, chips, optimizer
updates, protected-source scoring or terminal alpha are used. Overlap with the
active Native fit is retained in its wall-time interpretation.

[Source counts](report.json) · [Descriptive score breakdown](existing-score-overlap.json) ·
[Tokenizer inventory](tokenizer-inventory.json) ·
[Frozen census scope](../../../docs/PRIMITIVE_EXPOSURE_CENSUS_20261030.md).
