# TRAIN-only local arithmetic assertion coverage audit

A new, frozen selector inspects the first10000accepted TRAIN code documents.
It admits source-authored integer assertions derived from preceding constant
local assignments, including function/method scopes. Unknown statements clear
assumed bindings; source code is never executed. At least two expression-tree
arithmetic operations and an original verified answer are required.

9997documents round-trip through the tokenizer exactly;8500pass the bounded
Python parser. **Zero assertions qualify.** No evaluation panel, model score,
fit, new generated label or held-out read follows. The audit takes40.019seconds
on CPU and uses zero chips.33focused parser tests pass in0.12seconds; their
synthetic strings are software fixtures only.

This differs from the earlier top-level local-function-call selector, whose
zero result remains unchanged. Both are strict coverage lower bounds. Failure
to match constant-local arithmetic does not establish that the corpus lacks
recoverable dependencies, that imported functions are unreasonably difficult,
or that a model cannot learn computation. Neither audit justifies selecting
model-favorable examples or changing the active broad-NLL experiment.

[Plan and source hashes](plan.json) · [Complete counts](report.json).
