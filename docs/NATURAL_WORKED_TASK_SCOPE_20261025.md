# A conservative first task from original worked examples

This is a prospective feasibility restriction before any student model is
scored or trained on the proposed task. It does not change the frozen200-
document audit sample. Preserve all its candidates and labels,including hard
and ambiguous cases.

For the initial task,look for original open-ended quantitative word problems
with exact integer or rational conclusions and at least two source-authored
worked transformations. Allow addition,subtraction,multiplication,division,
percentages and ordinary unit conversions. Exclude transcendental calculations,
unresolved diagrams,missing statements,ambiguous temporal reference points,
incorrect source work and problems requiring case facts introduced only inside
the hidden bridge. Mathematical definitions and universal unit conventions
must be distinguished from case-specific facts.

Start with whole source documents of at most960 original tokens,leaving64
positions within the existing1024-token context for fixed formatting. Longer
pages need a separate segmentation audit and are not counted as semantically
invalid. This limits implementation ambiguity;it is not an architecture or
context-length change. No source text is truncated into an apparently complete
problem. The original ordinary-text stream keeps its full target coverage.

Keep the complete original premises,source intermediate spans and a compact
contiguous original conclusion. Preserve byte/character offsets and content
hashes. Never repair a missing premise,write a replacement explanation,or
create the answer using a calculator. A calculator or bounded symbolic parser
may verify an existing source claim;its outputs never become labels or steps.
The old repository's arithmetic checker is relevant;its generated-postorder
trace helpers are not used.

Exclude direct answer declarations from the visible context at every hiding
stage. Naturally occurring MCQ answer options are not automatically leakage,
but belong outside this initial open-ended scope. A title that states an answer
cannot stand in for missing premises. Equivalent units and exact rational forms
need a declared checker before any evaluation. Source correctness alone does
not prove that the hidden steps are recoverable from the visible context.

Automatic token-length triage is reported separately from complete agent
inspection. Neither is independent human verification. No population precision
estimate is inferred from reviewing the shortest documents first. Inspect the
remaining eligible candidates and establish reproducible extraction/verification
before proposing the curriculum schedule and its one-pass controls.

No new training,inference-depth change,held-out panel access or terminal
confirmation is admitted. A domain-specific finding would remain separate from
the broad natural-text NLL objective and must survive relevant resource controls.
