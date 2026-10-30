# Source-only coverage of authored Python interpreter examples

The previous local-function assertion census found six unverified, context-sized
candidates from one source group. It did not support a code-computation panel.
This finite alternative examines a distinct existing source format: doctest
prompts and their source-authored outputs. No source program is executed, no
answer is generated or replaced, and no model prediction enters selection.

Inspect the same first 10,000 accepted TRAIN code documents as the prior census,
with exact tokenizer roundtrip and the existing protected-source exclusions.
Isolate every original string literal containing doctest prompts. Count
single-line authored bounded-integer and boolean outputs. Separately count
syntactically closed examples using a fixed primitive/AST allowlist, with
transitive arithmetic-operation counts through straight-line assignments.
Reset tracked bindings after unknown statements, imports, attributes, shadowed
primitive calls, expected exceptions, doctest options, separate literals, or
gaps between examples. Syntax closure is not correct execution: intentionally
wrong-output tests must remain counted as UNVERIFIED candidates.

Report every funnel count and number of source groups with unverified multistep
outputs. A character-length count is descriptive, not proof that all context
fits 512 tokens. Do not form, score, tune on, or admit an evaluation panel from
these counts. An interpreter, exact original-span alignment, source/expression
isolation and adequate coverage would all be needed before a separate panel
proposal. This census can end in insufficient coverage without a larger scan.

One CPU attempt, 10-minute cap, 8 GiB RSS, original 512 GiB host, 35 GiB disk
and inode reserves. Zero chips, optimizer updates, held-out reads, model scores
or terminal alpha. Record concurrent host work; the original 16-chip ordinary
control continues unchanged. Private documents and source identifiers are not
published. Software fixtures are numerical/parser tests only.
