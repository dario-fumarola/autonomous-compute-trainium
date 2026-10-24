# Does the inexpensive connection improve learning by itself?

Development identity: `reinjection-pilot-v1`. This is not terminal confirmation
and does not change the completed recurrence comparisons.

Compare ordinary pretraining with the trained reinjection-only control:
one application of each of20 physical blocks, with the latter adding the
two-layer prelude state after its16-layer core. Both have150,436,608 unique
parameters and the same estimated training matmuls. Count the extra elementwise
addition in measured runtime. No teacher, generated research data, RL, additional
attention/MLP call, gating, normalization change or new prediction target.

Hypothesis: the inexpensive boundary connection itself improves held-out natural
text prediction. A positive result would justify it as a necessary architectural
control for recurrence. It would **not** establish useful additional depth or
explain the entire old recurrence advantage: old fits have different precision
histories, and a subsequent common-precision fixed-recurrence arm would be needed
for that attribution.

Require the bounded native qualification and complete-state recovery before
any tuning or scientific update. Use FP32 throughout from random initialization,
primitive SDPA, FP32 masters/moments, the qualified unscaled fused AdamW(.9,.95),
matrix decay0.1 with no norm/embedding decay, clipping1, context1024 and global256
packed windows (32 ranks, local batch1, accumulation8). No dataset wrapping.

Use the existing audited6.5B source stream with data seed23 and stateless routing
seed271828. Within each initialization seed, source windows/order and initial
parameter digests must match. The control is explicitly marked in every model
signature. No additional parameter is introduced. Independent CPU scoring uses
separate projections/attention/RoPE and the correct explicit boundary equation.

## Selection and diagnostic endpoints

Give **each** arm the same four peak LR candidates:
3e-4,6e-4,1e-3,1.5e-3. Tune at update1152 (approximately301M real targets), using
initialization seed11,2.5% token warmup and cosine decay over the final10%.
Select on the existing consumed TUNE cohort, weighted60/20/20 web/math/code.
Choose the lowest native FP32 NLL; when within0.0002 nats, choose the lower LR.
CPU-audit each selected TUNE readout before proceeding. Report the entire grid,
including edge selections and failures. Do not tune separately for the main seed.

Futility gate: if the selected cheap model's independent CPU score is more
than0.02 nats worse than ordinary on TUNE, stop before the main fits and preserve this development
negative. No unfavorable LR or seed disappears from the ledger.

Otherwise train both selected recipes from random initialization seed419 to
update3840 (approximately1.005B real targets). This seed is fixed before any
new tuning result. Warm up over2.5% of final source targets, retain stable
readouts at384 and1152, and decay from3456 through3840. Every phase restores
complete state exactly and retains the existing2% local FP32 gradient gate,
finite updates, identical final replicas and immutable source counters. Future
arithmetic across process restarts is not promised bit-identical.

Read the existing consumed DEV curve cohort, all real targets with60/20/20
domain weights. The main descriptive contrast is ordinary minus cheap NLL at
the **completed-decay** endpoint; intermediate points are stable curves. Whole
DEV independent FP32 audits must agree with native within0.0002 nats, and full
optimizer/model copy recovery must pass. Report domains and wall/estimated work.
One pair does not establish seed significance or a terminal0.01-nat useful gain.
No existing or fresh TEST is accessed. The new confirmation reserve stays sealed.

## Measured budget and ownership

Freeze the exact source/panel/packing hashes, prefix counts, selected-stage rules
and measured budget before tuning. Budget from the new control's100 warm update
rate and the qualified ordinary rate, with195s per training launch,100s per native
readout,30% margin and a30-minute CPU tail. Refuse launch if it no longer fits.
Training cutoff18:00 UTC and hard cutoff19:00 UTC on October24.

Run tuning on half1 while the separately registered5.1B ordinary control owns
half0. Once that owner closes, half0 may take remaining predeclared trials from
the same fixed queue (ordinary four LRs, then cheap four LRs), independent of
observed scores. The conservative budget assumes sequential tuning. Main
ordinary uses half0 and main cheap uses half1, after both selected TUNE audits. Source exposure is aligned, not wall-clock start time. All
overlapping job receipts, qualification, tuning, discarded branches and CPU work
remain separate costs. Failure closes this identity honestly; it does not relax
numerical gates or silently restart a phase.
