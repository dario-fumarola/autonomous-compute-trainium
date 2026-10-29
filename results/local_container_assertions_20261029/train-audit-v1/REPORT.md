# No eligible local list-tracking assertions in this TRAIN prefix

A separately registered audit inspects the same 10,000 accepted TRAIN code
documents as the arithmetic coverage study. It finds zero eligible original
list-length assertions. No evaluation panel is created and no model is scored.

The fixed verifier accepts locally initialized literal lists, at least two
length-changing append/extend/pop operations, and an original `assert len(xs)
== integer` target. The computed length only verifies the source-authored answer;
it never supplies a replacement label. The answer must be at least two, absent
from the dependency span as a numeric literal, and all premises must fit the
original contiguous 512-token context. Unknown calls, aliases and branches
invalidate local bindings. Explicit rebinding of `len` is excluded.

These restrictions provide a conservative static check. They do not establish
whole-program reachability or exclude arbitrary runtime monkey-patching. The
zero result does not show that natural programs lack recoverable computations;
it shows that this restricted selector cannot supply a panel from this prefix.
The independent syntax census found many integer-result assertions, often
involving calls and values that these local verifiers cannot resolve.

All 56 focused verifier/census checks pass, including aliasing, invalid
mutations, wrong answers, copied answers and explicit builtin shadowing. Test
fixtures are software checks and never research examples. The audit takes
43.29 CPU seconds, with zero chips, updates and held-out reads. It preserves
all original data and targets. No further selector widening or fit is admitted.

[Registration](plan.json) · [Complete aggregate receipt](report.json) ·
[Syntax census](../../local_arithmetic_assertions_20261029/syntax-census-v1/REPORT.md).
