# Authored code target coverage

The broader TRAIN-only syntax inventory finds 10,522 assertions in the first
10,000 accepted code documents, including 6,301 equality assertions and 1,859
with a literal integer expected value. Of 779 named-call integer assertions,
only nine refer to a preceding local top-level function definition. Eight use
literal arguments; six fit the 512-token context. All six belong to one source
group and involve loops, branches and indexing.

These are **unverified syntax candidates**, not correct programs, training
examples or a transfer panel. No source program was executed. The result does
not support building a new semantic benchmark from this small cohort. It also
does not imply that natural code lacks computational dependencies: imported
implementations, nonliteral arguments and cross-file context are outside this
inventory. The earlier strict zero-coverage audits remain unchanged.

Three parser fixtures pass, including an intentionally wrong expected answer
to ensure the inventory is never confused with semantic validation. Runtime
36.136 CPU wall-seconds; zero chips, updates, model scores or held-out reads.
Private source spans stay private. [Aggregate evidence](report.json).
