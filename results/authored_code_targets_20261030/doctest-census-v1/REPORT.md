# Interpreter-example coverage is insufficient under the bounded grammar

The first 10,000 accepted TRAIN code documents contain 2,184 parsed interpreter
examples in 423 original string literals. There are 179 authored boolean outputs
and 104 authored integer outputs. However, 2,127 inputs require unsupported or
unresolved context; only 57 fit the conservative closed-input syntax. No closed
literal-output multistep candidate survives, and no evaluation panel is created.

This is a limit of this extraction rule and cohort, not evidence that natural
code lacks computation. Library objects, imported implementations, methods and
other dependencies are deliberately not treated as available implementations.
No source program is executed or expected answer verified. All candidate counts
remain syntax evidence, including the intentionally wrong-answer software test.

Five parser tests pass. The original corpus order and tokenizer are unchanged;
9,997 documents roundtrip exactly and 8,500 parse within the bounds. The other
exclusions remain in the aggregate receipt. No held-out group, prediction score,
training update or scientific fit is involved. This result does not admit an
expanded source hunt or a hand-selected task panel.

Runtime: 38.626 wall seconds, 48.122 process CPU seconds, 1.177 GB peak RSS,
zero assigned chips. This small CPU census overlaps the unchanged all-chip
ordinary control; shared reservation opportunity time is not double-counted.

[Counts and costs](report.json) · [Frozen plan](plan.json) ·
[Protocol](../../../docs/AUTHORED_DOCTEST_CENSUS_20261030.md).
