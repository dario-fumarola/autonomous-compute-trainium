# Strict authored-code feasibility audit

The first 10,000 accepted TRAIN code documents yield **zero qualifying
assertions** under the frozen strict rule. Of these, 9,997 round-trip exactly
through the tokenizer and 8,500 parse as bounded Python; 1,497 do not parse and
three fail token roundtrip. No evaluation panel is created.

The rule requires a top-level source-authored integer assertion after a local
pure function, at least two arithmetic operations, no expected integer copied
from its literal arguments or function constants, and all premises inside the
512-token context. The checker never executes source code or generates labels.

This narrow rule is not a practical panel source in the sampled documents.
It is not evidence that the corpus contains no recoverable computation. No
model scores, held-out sources, optimizer updates or chips were used. The
bounded audit took 36.28 seconds. Parser fixtures pass 22 tests.

[Frozen rule and inputs](plan.json) · [Complete counts](report.json).
