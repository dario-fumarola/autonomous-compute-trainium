# Does authentic calculation exposure help generated computation?

This is a new, bounded learning diagnostic, not a continuation or confirmation
of the billion-target result. It may be admitted only after that comparison
and its exact readouts close. Keep its result regardless of sign. Do not change
the active scientific controller or automatically adopt an optimizer candidate.

## Question

The existing billion-target models solve few elementary calculations, and most
of the latent model's harder-answer NLL advantage occurs after the first answer
token has been supplied. The original128M prefix contains only617 intact
verified multistep calculations. A separately prepared authentic-document index
contains4,509 at approximately the same total exposure, plus9,816 elementary
calculations. This motivates an exposure test; it does not diagnose a proven
cause or establish a minimum model size.

Use a two-by-two design: original versus calculation-enriched natural text,
crossed with generated persistent states versus the strongest tested ordinary
shape. Both architectures receive the same source-selection opportunity. This
distinguishes a general benefit of better task exposure from an extra benefit
of the generated-state computation. Do not compare a specially selected latent
corpus against an ordinary model denied that corpus.

## Fixed conditions

Four fresh fits use one paired initialization seed730283 and1,957 updates each
(about128M real targets; publish exact counts). Within each architecture the
initial model must be byte-identical across source conditions. Each starts from
random weights. Retain the existing51.747M generated-state model and51.179M tied
eleven-layer ordinary model, context512,64ranks, local batch2, uniform real-token
NTP, AdamW lr0.001/decay0.1,64-update warmup, and complete final10% cooldown.
These inherited hyperparameters are not newly tuned for the selected corpus.
Keep the existing full-gradient balanced K2/K3 numerical solver schedule for
the latent model; K is not a supported thought-budget curve.

The original condition uses indexed-natural-sources-v1. The enriched condition
uses calculation-exposure-index-v1-recovery-v1: unchanged60/20/20 web/math/code,
half of math tokens from original documents containing verified calculations,
and half from the remaining mathematical documents. Every training target is
an authentic source token with ordinary uniform weight. No answer upweighting,
teacher, generated problem, supplied thought or repeated document ID is added.
Previous research reuse of these sources is disclosed. Packing differs between
source conditions, but is identical between architectures within a condition.

Run all four conditions regardless of intermediate quality. Retain pre-cooldown
and final full states, mature independent128-window CPU gradients, optimizer/RNG
restoration and64-replica checks. Use the existing unmodified reduction/clipping
path. Numerical family tolerance remains2%; no speed optimization is imported.
Audit windows follow every fit prefix and are not selected by loss.

## Readout and interpretation

After all four fits pass, score the unchanged384-window broad DEV panel and all
259 source-separated, TRAIN-expression-excluded harder arithmetic cases.
Use exact sequential FP64 latent evaluation with independent prefix checks.
Report every2/3/4–6-operation bin, first-answer-token NLL, full-answer mean NLL,
and exact answer-prefix accuracy. No new source slice, generation prompt, answer
prior or best-of-samples choice is introduced. Both fresh reserves stay sealed.

Report these predeclared contrasts with paired source bootstrap intervals:

1. Enriched minus original improvement for each architecture.
2. Latent minus ordinary benefit within each source condition.
3. The interaction: latent exposure improvement minus ordinary exposure
   improvement. A positive interaction means this source change benefits the
   latent model more; it does not by itself show that latent is the better model.

First-answer NLL matters separately: improvements confined to later
teacher-forced answer tokens do not establish stronger answer computation.
A promising mechanism screen requires the enriched latent model to improve
both first-answer and full-answer NLL over enriched ordinary by at least0.03
nats, with positive conditional lower bounds; its first-answer improvement
over original latent must be at least0.03, its first-answer interaction must
be positive, and its broad NLL may be at most0.005 worse than enriched ordinary.
Publish accuracy even if this gate passes. These are development decisions,
not final significance or a reliable-calculation claim.

One fresh seed on reused DEV cannot estimate training-seed variance. Equal
updates/source exposure are not matched training compute: count the existing
roughly3.78x latent work multiplier and all actual chip costs. No longer work
control, trained-pause fit, confirmation, size increase or corpus search is
automatically admitted. If both architectures improve similarly, the supported
finding is source-exposure benefit, not a special thinking mechanism.

## Resources and admission

Use four sequential all16-chip allocations with exclusive ownership, a4-hour
whole-queue cap,30-minute Native cap per fit,20-minute independent CPU audits,
and one60-minute exact readout phase. Require512GiB host reserve,192GiB worker
cap,35GiB free per output filesystem plus6GiB for the complete new queue, and
projected snapshot inodes above10,000. Freeze all code, source and panel hashes
at admission. Preserve every stopped attempt. No deadline renewal or automatic
retry. This document is a prospective design until an explicit plan is written.
