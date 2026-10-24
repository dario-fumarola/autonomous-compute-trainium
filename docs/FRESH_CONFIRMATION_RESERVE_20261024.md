# Fresh natural-source confirmation reserve

Data-only registration, independent of the active3B comparison. No model
scores, terminal attempt or alpha allocation are created by this preparation.
Current training and its DEV panel remain frozen.

Exclude every accepted document across all four corpus generations, all prior
protected source groups and the newer consumed TUNE/DEV/TEST panels. Retain
the original host/repository80/10/10 rule and the same source/license/length
filters. Among unused held-out groups, a fixed group hash assigns new
TUNE/DEV/TEST roles. Content-hash subsampling and deterministic receipt/row
order select candidates without inspecting a model's loss. Each domain/role
needs at least250k real targets and100 source groups, at most two documents
per group. No source generation or external teacher is used.

The existing full-corpus Python sketch index would threaten the training
memory reserve at this scale. The new method maps sorted accepted-ID arrays,
freezes a small candidate index, and streams old raw sketches against it.
Distinct old identities, not raw repeated appearances, determine crowding.
The same32 minhash values/eight bands, bottom256 Jaccard0.8 and256-neighbor
cutoff apply. Prior and newly admitted neighbor counts are combined. Every
submitted worker result must be consumed before trusting reconstruction
coverage; a shared coverage bit alone cannot prove that its matches were
integrated. Numerical CPU tests compare the inverted rule against the prior
index and cover lookup edges, duplicates and combined crowding.

Run eight low-priority CPU workers for at most110 minutes, then a bounded
independent audit. Keep512GiB host memory plus the entire remaining native
container allowance and35GiB disk. Require20GiB conservative allowance at
startup; yield before it falls below4GiB. No corpus-sized replicated Python
ID/sketch index is used. Record measured parent/reaped-worker CPU seconds,
failed attempts and wall intervals separately from overlapping training.

Qualification requires all accepted IDs to have reconstructed raw sketches;
independent SQL checks of exact IDs and source groups across every generation;
mapped-ID equality and coverage; unchanged prior panels; and independent
raw-row/full-token reconstruction for every selected document. The auditor
does not repeat every minhash comparison; that remains a disclosed approximate
screen with a separately tested algorithm and full prior-ID coverage.

Raw texts, tokens, provenance locators and identifier arrays remain private.
Only source receipts, code hashes, counts, qualification and cost aggregates
are public. Reserved TEST receives no model scores until a future candidate,
controls, fresh seeds, sample size and analysis have been registered. This
data job neither claims a positive result nor substitutes for that experiment.
