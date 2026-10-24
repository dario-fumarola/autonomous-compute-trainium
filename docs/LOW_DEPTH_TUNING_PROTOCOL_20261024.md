# Low-depth three-arm LR screen

Gate launch on complete10-case model/precision/training qualification and the
finite new-driver checkpoint/branch integration pass. Use the prepared scientific
seed11 initial weights, never qualification/trained states. Keep natural source
order and each global512-window objective aligned. Context1024,32 ranks per
eight-chip job, accumulation16, FP16 body/FP32 head and masters, manual8192
scale, eager FP32-polar Muon plus fused unscaled AdamW. All16 chips can run two
independent fits; independent CPU scoring overlaps on the same host.

Each arm receives the same four opportunities: peak0.002,0.004,0.008,0.016 for
both optimizer families at a1:1 ratio. The pinned release's central rate is0.008.
Each runs576 updates, the same approximately301M real targets. Warmup is an
absolute50M targets, then stable to update288, then cosine cooldown to zero at576.
Cautious Muon decay stays0.2 on the stable prefix and decays linearly to zero
during cooldown. Momentum warms0.85→0.95 with the same source warmup. These
schedule/precision choices are disclosed deviations from the release, common
to all three arms. Numerical qualification used a conservative rate; it does not
select the scientific LR.

Use ordinary, true one-pass Parcae, and per-sequence Poisson(lambda2)/minimum1/
tail1 Parcae. Primary candidate readout is depth2, controls depth1. Exact same
unique family parameters and initial weights for one-pass/recurrent; ordinary
has the declared smaller machinery inventory. Frozen sampled depths preserve
the original per-sequence work and full head/attention/replicated optimizer costs.

The first20 updates form a finite source segment, saved with complete state;
the remaining556 restore that exact state in a fresh process. This is counted
scientific tuning work from its own random initialization, not reused
qualification work. Each endpoint finishes cooldown. Primary scores use pinned
upstream FP32 CPU model code on the existing consumed TUNE cohort, scoring every
real source token once. No TEST or fresh reserve is available to this controller.
Choose the lowest weighted60/20/20 NLL independently per arm, with lower rate
breaking an exact tie. Never start main fits from these tuning weights.

All trials and CPU evaluations are planned in advance. Count12 actual trial
paths and all failed/early-stopped attempts. Equal opportunities/source exposure
do not mean identical tuning work: publish each arm's executed work and the
entire screen cost. Use measured100-warm-update rates to justify the budget,
including cold launches, checkpoints and CPU tails separately. Full screen cap
3hours, each native phase25minutes and CPU score15minutes; latest start
00:30UTC October25. No automatic increase in grid, exposure or budget.

A nonfinite-loss/gradient trial is a recorded unsuccessful LR opportunity;
other predeclared rates may finish. Compiler, source, numerical, counter,
recovery or reserve failures stop this execution for diagnosis. Routine
cancellation owns only the affected workers; shared emergencies retain the
instance-wide reserve rule. Explicit distinct non-ephemeral root communication
ports are registered per phase.128GiB per worker/256GiB paired,512GiB host and
35GiB filesystem reserves. Final main-budget endpoints and selected-LR numerical
checks are a separate registration after the screen; the research goal remains
active regardless of the selected rates.
