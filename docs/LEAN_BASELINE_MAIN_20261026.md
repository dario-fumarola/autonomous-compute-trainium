# Stronger-control paired development comparisons

This follows qualified complete three-rate selection for corrected FP32-adapter
cheap mixing and conventional ordinary8. Lean One inherits0.032 after the
qualified head-removal parity proof. No new objective,corpus,precision or model
search. Retain upper-boundary tuning limitations.

Compare all three controls with their same seed's existing audited fixed-two
2B endpoint. Use seeds5231 and17293 and their original random body initialization
and state_seed=seed+2009. Ordinary8 has its own random eight-layer initialization;
all parameter counts remain below or equal to141,963,840. Qualification-updated
weights are excluded. Each model starts anew,with50M-target warmup and its own
half-budget cooldown to zero. Never extend an already cooled checkpoint.

The immutable complete-matrix budget is1,272,689,958,300,549,120 operations.
Lean One:4706 updates/2,463,002,486 targets;corrected cheap:4674/2,446,254,768;
ordinary8:5275/2,760,803,936. Charge output projection,attention,all executed
mixing/core/backward work and all32 replicas' optimizer matrices. Work endpoints
are floors,short of the target by at most0.018%. Report chip-time separately.
The accepted6.5B natural stream covers all endpoints without wrapping.

Run both seeds of each control concurrently,using all16 chips. This changes
only the unlaunched prototype's scheduling:all six scientific fits,initializers,
endpoints,rates and decisions remain unchanged. It avoids leaving half the
instance idle while the third model in a seed completes. Order:corrected cheap,
ordinary8,lean One. Each paired control has a separate four-hour complete cap
based on qualified warm-update measurements,with own storage/inode admission.
This permits maintenance between controls without renewing a stopped stage's
clock. A stopped stage requires explicit preserved recovery;no automatic retry.
Both seeds belong to every control;do not select the favorable one.

Lean One starts with a counted two-update state/counter readback and fresh-process
restore. Mixing/ordinary8 have the identical scientific-engine counted restore
checks from completed tuning. All main endpoints require complete replica/state
checks,full512 next-TRAIN gradient gates against pinned independent FP32 model
code,and independent complete consumed-DEV scores using window-position-only
noise. Unexpected numerical failures stop owned workers and preserve the first
error and all costs. Do not certify old failed cheap training retrospectively.

Report each control's weighted NLL and recurrence advantage plus paired
source-group uncertainty;these intervals are conditional on a training seed.
Two reused development seeds and this consumed panel are not confirmation.
If recurrence comes within0.01 nats of any qualified cheaper challenger,do not
claim a useful advantage over that challenger. Preserve a narrower same-family
result when appropriate.4B is a separate conditional decision,not an automatic
continuation and not a predicted0.02 gain. No TEST,fresh reserve or terminal
alpha is admitted here.
