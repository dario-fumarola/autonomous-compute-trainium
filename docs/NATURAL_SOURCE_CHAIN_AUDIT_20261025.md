# Can original equation chains give a cleaner computation task?

The word-problem inspection exposes missing givens,ambiguous conventions and
incorrect work. Investigate a narrower source format before any student score:
an original fully numeric expression,one or more original intermediate
expressions,and an original scalar conclusion connected by equalities.

Unlike a word problem,the numeric expression contains all case-specific inputs.
A bounded exact-rational checker can verify every existing equality. It never
writes a replacement answer or generates intermediate work. This is natural
procedural supervision without a teacher model,not supervision-free discovery
or unchanged raw-text pretraining. Preserve the word-problem audit separately.

Freeze the checker and extraction rules before counting:at most240 characters
per expression,six binary operations,63 AST nodes,integer exponents bounded
by8 and exact values bounded by10^12. Support explicit arithmetic and a small
fraction/operator TeX subset. Reject functions,variables,malformed expressions,
unequal values and unsafe syntax. Require an original intermediate with fewer
operations than the premise and at least one operation still remaining. Reject
cases where the answer is already a numeric literal in a retained expression.
Keep the first eligible chain in source order,at most one per document.

Audit the existing200-document inspection sample and the same20,000 accepted
TRAIN mathematics documents already recovered by the earlier audit. Reconstruct
only their matching URL/content-hash identities from the same pinned six raw
shards. Use at most4 CPU readers and600 seconds,with the existing memory/disk
reserves. The earlier lexical sample is deliberately not a population sample.
Record source spans,UTF8 hashes,verifier status,operation counts,source-group
diversity and exact-expression duplication. Publish aggregates only. Do not
open DEV,TEST,reserve data or invoke a student/teacher model.

Only source-span candidates are produced. No curriculum inputs,generated
problems,solutions or labels enter training. Arithmetic strings in unit tests
are parser software fixtures and never become research examples. Inspect
candidate sources before accepting an extraction dataset. A numerical chain
may be a tractable first mechanism task;it is not a general reasoning or
broad-text improvement claim. Any fit still needs a separate data-view protocol,
strong one-pass/resource controls,measured costs and independent confirmation.
