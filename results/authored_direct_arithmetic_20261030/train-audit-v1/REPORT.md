# Authored direct calculations have usable source coverage

The bounded TRAIN-only audit scans 100,000 accepted natural mathematics
documents in unchanged order, with exact tokenizer roundtrips. It finds 3,212
verified source-written multi-operation equalities. Of these, 2,968 have exact
target alignment and a complete premise within 512 source tokens, across 954
source groups. They contain 8,149 authored answer tokens. Another 244 equalities
fail exact token-boundary alignment. No model has been scored.

| Arithmetic operations | Eligible calculations |
|---|---:|
| 2 | 1,668 |
| 3 | 812 |
| 4 | 257 |
| 5 | 171 |
| 6 | 60 |

The parser checks the complete existing arithmetic expression using exact
rational arithmetic, then verifies the existing scalar answer. That answer
must not appear among the expression's operand literals. It returns source
spans only. It neither creates labels nor removes source-written work. Unlike
the earlier worked-chain extraction, a direct equality needs no intervening
written solution step. All targets and surrounding text remain authentic.

These are coverage counts, not unique problems or a prepared evaluation set.
Expression deduplication, source separation, copy checks and a frozen readout
must precede model evaluation. The strict grammar omits many genuine examples;
it does not measure the total amount of computable text. In the original packed
training windows, 2,896 of these targets retain their entire parsed premise.

Nineteen numerical/parser fixtures pass. Cost: 329.217 CPU wall-seconds, zero
chips, updates or held-out reads. Both fresh confirmation reserves remain
untouched. [Full aggregate](report.json).
