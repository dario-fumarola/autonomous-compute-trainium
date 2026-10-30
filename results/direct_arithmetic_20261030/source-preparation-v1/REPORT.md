# Direct authored arithmetic: frozen development panel

The source-only preparation selects **353 calculations from 353 source groups**:
128 with two operations, 128 with three, and 97 with four to six. Their 1,048
answer tokens and every visible premise are unchanged source text. The panel
contains no supplied intermediate steps, rewritten questions or generated
answers. It excludes literal answers already visible before the expression.

The source groups occupy a hash bucket excluded from every indexed training
run. Both protected confirmation reserves remain excluded and unscored. Exact
literal-valued expression ASTs from the complete scientific mathematics training
prefix are excluded, leaving zero verified-expression overlap. This is a narrow
expression/source check, not a guarantee of universal semantic non-contamination.
Of the selected expressions, 284 have an operator-tree shape seen in training.

The source scan exhausts 47,957 eligible development-bucket documents, all with
exact tokenizer roundtrips. It finds 1,067 eligible calculations and excludes
95 matching training expressions. Fixed hash selection, one case per source
and expression, and a predeclared minimum of 32 per bin all precede model scores.

The actual 128,025,288-target scientific training prefix contains 19,004 math
documents and 681 verified equalities under this strict grammar. Of those, 617
retain the full parsed premise in the original packed context, contributing
1,697 aligned answer tokens. These counts measure this extraction rule, not
all arithmetic or computation in the corpus. They are exposure evidence, not
an explanation established for any model outcome.

The later panel helper conservatively excludes leading-sign expressions that
could be numeric tails of symbolic formulas and removes a sentence-final period
from the answer span. The earlier TRAIN coverage audit remains unchanged; its
broader counts are not this panel's counts.

Preparation costs 274.720 CPU wall-seconds, zero chips, updates or model scores.
Both fresh reserves remain sealed. The separately registered readout evaluates
all three saved seeds and both controls; this preparation admits no training.

[Exact preparation receipt](report.json) ·
[Source protocol](../../../docs/DIRECT_ARITHMETIC_DIAGNOSTIC_20261030.md) ·
[Readout protocol](../../../docs/DIRECT_ARITHMETIC_READOUT_20261030.md).
