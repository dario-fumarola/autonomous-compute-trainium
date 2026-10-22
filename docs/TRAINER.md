# Training and qualification

`scripts/train_distributed.py` runs one fixed-shape phase. It uses the same
packing, loss normalization, clipping, optimizer and schedule for timing and
training. Launch through `scripts/supervise.py`, which invokes the immutable
native launcher. The supervisor checks host memory and both output filesystems
every two seconds, detects compiler-error logs, tracks container OOM events,
and enforces a wall-time bound. A report plus a clean supervisor exit is required.

Packing retains short documents and tails. BOS/EOS/padding are masked targets;
the head still computes their fixed slots, and FLOPs include that work. Causal
attention can cross document boundaries within a packed window. DEV isolates
documents. The deterministic stream cursor and actual domain counts are saved.
Production data may not silently wrap. `--qualification-repeat` explicitly
permits repeating the small natural sample for throughput checks; those curves
are not evidence of generalization or new unique token exposure.

Use `--packed-corpus` for shared file-backed token/domain/count arrays rather
than decoding a full Python corpus in each worker. File hashes, context, seed
and source provenance are checked. `pack_corpus.py` materializes this contract
for the small JSON sample; a large-corpus freeze must build the same files with
streaming preprocessing. `--dev-corpus` can pin a separate DEV-only file, so
evaluation does not read the complete training corpus. Large-source preparation,
I/O timing and duplicate auditing remain launch gates.

For accumulation, each microbatch produces a summed real-token loss. Its
backward scale is world-size / global real-target count for the whole update.
DDP averages gradients. All but the final microbatch use `no_sync` around both
forward and backward. Counts come from the immutable global stream, using
prefix sums rather than an extra device collective. FP32 masters and Adam
moments remain FP32 under BF16 autocast; full-vocabulary NLL is FP32. Global
gradient clipping and a finite-norm check precede each optimizer update. The
norm check synchronizes and is included in measured throughput.

Every new shape/precision/phase validates loss and full gradients against CPU
before DDP updates. Predeclared tolerances are 0.003 nats / 2% relative gradient
L2 for FP32 and 0.03 nats / 8% for BF16 against CPU with the same autocast.
These are correctness tolerances, not precision equivalence claims; compare
research arms using the same selected precision. The small 64-rank global
gradient check also validates variable masks and accumulated reduction.

Atomic checkpoints include complete model/optimizer, schedule, CPU RNG, stream
cursor, source/config hashes, real-token exposure and cumulative matmul FLOPs.
Current models have no dropout or device random sampling. A stochastic model
would additionally require per-rank device RNG recovery. Checkpoints are mapped
read-only when loaded to share file-backed pages across 64 ranks. CPU allocation
arenas are returned before compiling the next graph. Full state digests validate
recovery on CPU and after device transfer, and final full replica states agree.

`scripts/run_phases.py` executes a concrete JSON plan. Every phase exits before
DEV evaluation (`scripts/evaluate_checkpoint.py`) and before restoring the next
phase. This provides periodic checkpoints/evaluation without co-resident large
training and inference graphs. A pause switch therefore cannot retain the old
graph. Any worker, numerical probe, supervisor or DEV failure stops the plan.
All phases must use a consistent global batch to preserve data-order pairing.
The controller sets explicit container caps at process boundaries: the checked
tiny plan uses 160 GiB, the measured 150M plan 256 GiB, and single-core DEV
64 GiB. The host preflight reserves the actual cap plus 512 GiB. Larger models
or datasets need their own measured memory limit; do not inherit an unrelated
compile cap for a small evaluation phase.

Stable checkpoints can fork independent cosine-decay branches. Set `branch:
true`, the chosen `resume` checkpoint, and decay boundaries in that phase's
plan entry. Its decay must start at the stable checkpoint's actual token count.
The controller retains its stable continuation pointer, while evaluating the
branch's checkpoint. Peak LR and warmup cannot change; a decayed checkpoint
cannot masquerade as a stable trunk. Compare only endpoints whose decay is
complete. Charge extra branch work separately, as in `performance_budget.py`.

The checked-in `configs/qualification-phases.json` is a tiny-model, natural-data
software check, using all 64 ranks and a small DEV subset. It is not a scientific
training plan. A scientific plan requires a separately frozen source/order and
near-duplicate audit, selected learning rates at the measured batch, completed
trainer qualification and measured budget. TEST remains sealed.
