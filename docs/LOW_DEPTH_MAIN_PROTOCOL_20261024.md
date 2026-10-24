# Finite small-core main pilot

This registration follows the four-rate301M screen and independent full-global
gradient checks of every selected tuning endpoint. It does not choose an
architecture or readout based on fresh held-out outcomes. Ordinary140.78M,
true one-pass Parcae141.96M and sampled lambda2/minimum1/tail1 Parcae141.96M
use new random seed1231 weights, with all common backbone tensors cloned and
the entire one-pass/recurrent initial state identical. No qualification or
tuning weights initialize a main fit. Precision, operators and512-window batch
are fixed by the complete numerical/throughput gates.

Each main path receives at most1,858,147,163,150,745,600 estimated dense matrix
FLOPs, including the tied output head, full-square attention, actual sampled
forward/backward counts and all32 replicas' Muon matrix operations. The exact
prepared whole-update endpoints are ordinary6967,one-pass6872,recurrent6370:
approximately3.646B,3.597B,3.334B original targets. Work shortfall from rounding is
below0.01% in every arm. Actual pointwise work, hardware tile padding,
communication, cold compilation, diagnostics and elapsed chip time are separate;
this is a matrix-operation match, not a chip-time match.

Each arm keeps its independently selected scalar LR and1:1 AdamW/Muon rate.
Absolute50M-target warmup and warmed momentum are shared. Hold the peak rate
and cautious decay through the declared midpoint, then cosine LR/linear decay
to zero at that arm's endpoint. Save the untouched stable update960 state and
fork a separate960→1920 cooldown to the common1,004,887,590-target endpoint.
The extra branch work is charged; no annealed endpoint resumes as a stable trunk.

The first main segment stops at1920updates, before the main path's cooldown.
Its next complete TRAIN gradient must pass the registered2% overall/family and
0.003-nat independent CPU gates before continuing. Repeat the gradient check
at each completed main and equal-data endpoint. These probes perform zero
updates and do not advance the model's source/optimizer counters; their full
diagnostic compute, duplicate TRAIN exposure and elapsed time remain costs.
Exact complete model/optimizer/RNG recovery, source/work counters,32-replica
equality and compiler/fallback checks are mandatory on every native segment.

Primary readout: candidate depth2 versus true one-pass Parcae at matched work.
Also report ordinary comparison, each independently annealed equal-data endpoint,
and candidate supported exits1,2,3. Pinned upstream FP32 CPU model code scores
every real target on the existing consumed DEV curve cohort. Stable576/1920
checkpoints describe learning progress; they are not substituted for annealed
endpoints. No fresh reserve/TEST readout or terminal alpha is available here.
Keep the original60/20/20 weights and0.01-nat useful-effect threshold.

This is one pilot seed: document uncertainty does not supply training-seed
significance. An apparent win advances only to the existing fresh≥5-seed
confirmation policy with the strongest eligible one-pass/cheap-connection
challenger, full resource accounting and unchanged alpha spending. A within-family
gain alone is not practical superiority. Preserve negative/inconclusive outcomes.

The small-core charter's natural-code distant-context diagnostic remains a
separate prespecified mechanism readout. It requires both positive full-context
extra-pass gain and positive full-versus-short interaction at supported exits.
A favorable pooled NLL result does not establish that mechanism automatically.
One same-forward-distribution full-gradient follow-up is reserved only under the
charter's advancement rule; no simultaneous router,objective,size or depth search.

Use two independently owned eight-chip jobs and release native workers during
CPU tails. The fixed queue must finish every registered stage or write an
explicit failed prerequisite. Main execution has an8hour hard cap, each native
phase at most200minutes, and each CPU score at most20minutes. Measured warm
training predicts approximately4hours of whole-instance reservation including
extra branches; cold launches,numerical audits,evaluation and recovery fit inside
the separate cap. The launch plan records exact measured estimates. A resource,
numerical,source,compiler,recovery or deadline failure stops advancement; routine
cleanup owns its workers, while genuine instance reserve emergencies retain the
shared stop policy. Keep512GiB host and35GiB disk reserves,128GiB per native job.
