# Fixed recurrence at a more mature exposure

Registered after the completed variable-depth study, before any3B update of
the revised recipe. This is a bounded development comparison, not the final
independent confirmation in the ongoing charter. Existing TEST is consumed
and will not be scored again. No new architecture, learning rate or exit is
selected within this campaign.

## Question and fixed choices

Does the healthy fixed-two-pass recipe's equal-data benefit repay its extra
training work at3B real targets? The prior study establishes a three-seed
equal-data TEST gain of0.052743 nats at1B, but a descriptive matched-work
deficit of0.051258. The old recipe's3B result does not answer this question
about the revised common optimizer/warmup/LR recipe. Longer training is an
empirical check, not an assumption of eventual crossover.

Keep the150,436,608-parameter pre-norm model, P2/R16/C2, additive injection,
ordinary one pass versus fixed two, full backpropagation,32-rank eight-chip
layout, local batch4, accumulation2,1024 context and global256 windows.
Retain each arm's already selected peak (ordinary6e-4, fixed1e-3), AdamW
betas(.9,.95), eps1e-8, matrix decay.1, zero embedding/norm decay, clip1,
unscaled BF16, FP32 masters/moments, routing271828 and data order23. No
teacher, synthetic research examples, thoughts, RL, halting or separate norms.

Use seeds103/211/307, the same paired seeds of the development study. Restore
the untouched stable checkpoint at update3456 (904,398,582 targets), never
the annealed1B branch. Source migration changes only fingerprints to the
independently audited6.5B append-only descendant. Every earlier source
position, model/optimizer/RNG tensor and past work counter must be preserved.
CPU full-state copy/readback and native old/new-source continued-update
equivalence gate all longer training. Native qualification repeats are costs,
not scientific exposure or continuation parents.

## Endpoints and schedule

| Endpoint | Updates | Real targets | Stable-to-decay switch |
| --- | ---: | ---: | ---: |
| Ordinary, equal data | 11,520 | 3,014,637,826 | 10,368 |
| Fixed two passes | 11,520 | 3,014,637,826 | 10,368 |
| Ordinary, matched work | 19,490 | 5,100,221,387 | 17,541 |

Counts come from the frozen real-target prefix array, not nominal padded
positions. Ordinary/fixed step matmuls are286,044,821,913,600 and
483,956,914,913,280. Matched ordinary updates are the floor of11,520 times
their ratio; disclose the small realized work difference. The vocabulary
projection and backward pass are included, optimizer/communication/compiler
work are measured separately. Inference work also remains higher for fixed.

Warmup retains the original25,121,734 targets; each endpoint completes cosine
decay to10% of its peak over its last10% of updates. Ordinary forks its3B
decay branch from the stable10,368 checkpoint, then resumes that untouched
stable parent toward its matched-work endpoint. The3B branch is not part of
the matched model's history. Count its extra branch cost in the campaign.
Stable7680/10368 readouts are descriptive and cannot replace endpoints.

## Readout and decision

Score ordinary only at its trained one pass and fixed only at its trained two
on the existing DEV panel. Primary development contrast: matched-work ordinary
minus fixed NLL. Secondary: equal-data ordinary minus fixed. Publish every
seed, domain totals, nominal paired-seed95% and two-contrast Bonferroni family
intervals with the useful0.01-nat threshold. These are exploratory intervals
after previous research, not a fresh terminal significance claim. All three
pairs run to the declared endpoints; no stopping or endpoint substitution to
obtain a favorable score. Fail closed on numerical/resource/deadline errors.

Independently score all nine endpoints in FP32 on the full DEV panel, verify
full optimizer/model recovery copies and hashes, replica agreement, source
counts/order, initialization and past-state lineage. A positive mean alone,
or an unannealed crossover, does not achieve the ongoing goal. Further
exposure needs a new measured budget; this campaign does not authorize an
automatic10B extension. Final confirmation requires fresh held-out groups,
new independent seeds (at least five), frozen choices and the charter's alpha
allocation. No TEST is accessed here.

## Execution budget and recovery

Use both eight-chip halves concurrently. Queue ordinary103/fixed103 first,
then ordinary211/ordinary307/fixed211/fixed307 as halves become available.
Do not resize a scientific fit. Qualified100-warm-update rates are900,268
ordinary /577,029 fixed real targets/sec/job. Remaining training work predicts
3.61 node-hours; adding30% and90 minutes for launches, evaluation, copies and
independent audits predicts6.19 hours. This is a measured extrapolation, not
a promised completion time. The recorded plan fixes absolute training and
hard cutoffs before launch (10:15/11:00 UTC on October24).

Keep periodic full checkpoints and atomic receipts. Resume validates only the
saved dependency paths/hashes and exact successful launch arguments; unrelated
new helper files do not change a saved freeze. Completed stages are reused
with their original closure receipts. Incomplete native stages require an
explicit new recovery attempt from a validated full checkpoint, preserving
the failed identity; no silent restart or discarded source exposure.
Independent liveness/deadline monitoring must outlive the controller.

Report retained per-model histories, shared ancestors without duplicate
charging, extra decay branches, tuning, qualification failures, data prep,
compile/setup,100-update timing, CPU audit tails, idle tails and assigned versus
whole-node opportunity cost separately. Concurrent elapsed intervals must
not be added as if sequential. Existing full-state copies remain on-instance.
Keep512GiB host memory and35GiB output-filesystem reserves.
