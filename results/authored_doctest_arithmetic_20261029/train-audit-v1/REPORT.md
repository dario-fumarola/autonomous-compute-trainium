# Authored doctest arithmetic coverage audit

A frozen, CPU-only parser examines the first 10,000 accepted TRAIN code
documents in unchanged source order. Of these, 9,997 have exact tokenizer
round trips and 8,500 parse within the Python bounds. No authored example
meets the registered strict grammar: pure local integer operations in genuine
docstrings, at least two expression-tree operations, a matching canonical
source-authored integer output not already present in the dependency text, and
all premises fitting a contiguous 512-token context.

This is insufficient coverage for this diagnostic, not evidence that natural
code lacks recoverable dependencies. Imports, calls, collections, most library
examples, truncation and the strict non-copy/operation rules exclude many
authentic cases. No label is generated or substituted, no source code is
executed, no held-out group or model is read, and no scientific fit follows.
The audit takes 42.797 CPU wall-seconds, zero chips and zero optimizer updates.
The parser and related verifiers pass 43 numerical/software fixture tests.
