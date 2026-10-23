# Untied replication and resource control — separate confirmation follow-up

The seed-23 untied result is discovery evidence: 2.489247 DEV NLL, versus
2.518071 ordinary and 2.549316 shared at the same 1.005B real targets. It uses
251.1M parameters, compared with 150.4M. Preserve the original three-seed shared
negative and the one-seed discovery unchanged.

This follow-up is declared before either new seed is trained and before any
TEST score is opened. Complete the current 3B pair, its independent numerical
audits and trained-path DEV calibration first. Then allow at most four hours,
with an absolute ceiling of 23:30 UTC on October 23. Keep the same instance,
64 ranks, local batch 2, accumulation 2, context 1024, plain AdamW, FP32 masters
and moments, BF16 matmuls and clipping. Do not introduce another mechanism.

## Replicate the positive control

Fit untied initialization seeds 47 and 71. Reuse the already selected 3e-4 LR;
no new LR search or seed-specific tuning. Use the original data order seed 23,
identical source targets and boundaries, warmup over 40 updates, stable training
to update 3456 and the same 384-update final decay to update 3840. The seed-23
discovery is not retrained. Retain all models regardless of their scores.

For each added seed, verify that the initial two-pass function matches its
shared-core initialization and that its base initial weights match the original
paired ordinary/shared control. Existing native global-gradient, optimizer,
replica and throughput qualification covers this unchanged implementation;
every fresh phase still passes its CPU loss/full-gradient and recovery probes.
The untied control remains intentionally larger. A repeated win is not evidence
of a parameter-matched computation advantage.

## Match estimated training matmul work

Train an ordinary seed-23 endpoint at the largest whole update whose cumulative
estimated matmul FLOPs do not exceed the untied 3840-update endpoint. Compute
that update from the frozen per-update FLOP counts, not from quality results.
It is approximately 6496 updates / 1.70B targets. Report exact targets and the
small rounding shortfall. Count the tied vocabulary projection.

Fork the untouched ordinary update-3456 stable checkpoint, preserving every
Adam moment, RNG state, cursor and past-FLOP counter. Validate the complete old
packed prefix in the already qualified 3.25B append-only corpus. Keep peak LR
1e-3 and original warmup history. Start decay at floor(0.9 * endpoint_update),
then finish cosine decay to 0.1 of peak at the compute endpoint. Both compared
models therefore finish annealing. Readouts use the fixed natural-text domains
and weights. This matches estimated dense matmul work, not all optimizer,
communication, compilation or campaign costs; record those separately.

## Inference and statistical rules

Freeze all twelve terminal models before opening the existing sealed TEST:
the original six ordinary/shared 1B models, all three untied 1B models, the
ordinary/shared seed-23 3B pair, and the ordinary compute-matched seed-23 model.
The confirmation scope is expanded before TEST access; archive the original
scope and retain all earlier results. No projected or exploratory-depth model
is eligible. Exact checkpoint, trained-depth, evaluator and analysis hashes
must pass the release gate. No further model choice uses TEST.

The original three-seed comparison remains the primary confirmation. Report
untied paired differences for every seed, the combined three-seed descriptive
interval, and a separate fresh-seed 47/71 interval (Student-t, df=1). Seed 23
motivated this follow-up; do not present the combined set as an independently
preregistered discovery. Resource matching is one pair: source-group bootstrap
intervals condition on those fitted weights and do not estimate seed variance.
Use the fixed 0.01-nat useful-gain threshold, and keep token-matched and compute-
matched questions separate. Publish every domain and declared comparison.

Require a complete measured budget before launch: conservative training time
from the observed ordinary/untied rates, process/checkpoint/evaluation overhead,
and a protected final confirmation/reporting reserve. A prerequisite failure
or insufficient time produces an explicit result; it cannot silently change
seeds, optimizer, data or the endpoint. No synthetic corpus, teacher, RL,
learned halting, larger scale or partial-sharing variant is part of this cycle.
