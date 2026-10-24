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

## Memory repair, executionv3

Executionv2 stopped at its extra conservative memory guard during prior-raw
screening. Its pre-score10800-candidate pool had already been frozen. The
original pool hash, source-input plan, token/sketch arrays and prior-ID index
hashes are pinned in the [v3 reference manifest](../results/fresh_confirmation_reserve_20261024/v3/frozen-pool-reference.json).
The new execution reuses those exact candidates, rebuilds prior-ID indices,
and starts **new zeroed coverage/rejection maps**. No partial rejection state
or uncollected neighbor results are trusted.

The prior scan now keeps one Parquet reader open at a time and releases Arrow
allocator buffers at file boundaries. Candidate collection retains its
original interleaved order when used; this repair does not reselect the frozen
pool. Early skipping occurs only after definitive rejection. All non-rejected
candidates retain every old neighbor, and final admission remains in fixed
content-hash order. Changing the prior scan order therefore preserves the
acceptance rule; rejection-reason bookkeeping need not have identical order.

The [read-only resource probe](../scripts/profile_reader_memory.py) on the
first256 rows of the same16 pinned web files measured about853MiB RSS growth
with readers retained versus37MiB with sequential close/release. This is a
bounded resource measurement, not a full-corpus peak-memory guarantee.
Numerical CPU tests cover reader lifetime/row coordinates, pinned-pool tamper
rejection and order-invariant admission. The92-test suite passes.

Executionv3 uses12 low-priority CPU workers with26GiB conservative startup
allowance, yielding below4GiB while retaining the512GiB host reserve and the
remaining native-container allowance. Every prior raw identity is screened
again. The independent SQL, coverage and raw-row/full-token audit is unchanged
and remains required. [Registration](../results/fresh_confirmation_reserve_20261024/v3/registration.json).
No model has scored these new panels and no terminal alpha has been spent.
