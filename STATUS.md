# Status — 2026-10-25, 01:27 UTC

**Fresh-seed main training is running on all16 chips.** True one-pass and
recurrence have performed more than60 real updates; ordinary is waiting for
an eight-chip slot in the fixed three-arm queue. Main seed1231 starts from
fresh cloned initialization,not qualification or tuning weights. No main DEV
scores are available yet.

All three selected301M checkpoints pass independent complete512-window TRAIN
gradient checks: overall errors0.223%,0.257%,0.264% for ordinary,one-pass and
recurrence; the maximum loss difference is0.00000460 nats. The2% gradient and
0.003-nat loss gates remain unchanged. Every main1B/endpoint is checked again.
[Selected-gradient evidence](results/low-depth-gradient-selected-v1/report.json).

The frozen main paths have the same1.858147e18 estimated matrix-work budget,
including head,attention and all32 replicas' Muon work. Endpoints are3.646B,
3.597B,3.334B real targets for ordinary,one-pass and recurrence,with separate
completed-decay1.005B branches. Warm measurement predicts3.86hours of node
reservation; execution is capped at8hours. No fresh reserve or terminal alpha.
[Main plan](results/parcae_low_depth_pilot_20261024/main/plan.json).

All12 rate opportunities completed. Each arm selected0.016,the upper grid
boundary: TUNE NLL2.884985,2.950986,2.957179. Recurrence is0.006193 behind its
parameter-exact one-pass control here; the optima are not bracketed. These
shorter scores do not decide the main comparison.
[Screen report](results/parcae_low_depth_pilot_20261024/REPORT.md) ·
[Observed figure](results/parcae_low_depth_pilot_20261024/figures/tuning.png).

A read-only main-run sample found all64 logical cores active,with84.6% average
runtime busy time over11 accepted intervals. That is not tensor-peak utilization
or a campaign average. [Runtime evidence](results/parcae_low_depth_pilot_20261024/main-runtime-busy-sample.json).

A CPU-only randomness audit is registered before any main scores. The frozen
pooled evaluator derives its latent-state seed from whole-document identity,
which is a content hash. The audit will retain the same weights/targets and
use only original prefix-window position. It preserves the frozen readout and
does not select recipes,exits or seeds. No quality effect is assumed. The
natural-code intervention already uses position-only randomness.
[Audit registration](results/low_depth_prefix_noise_audit_20261025/plan.json).
The CPU suite passes172 tests,including prefix/future and length invariance.

Independent actual-size upstream operator counts agree exactly with the
inference ledger: at1024 positions,one family pass costs159.049GFLOPs and two
cost195.691GFLOPs,including the50.332GFLOP tied head. This is23.038% more forward
matrix work,not a measured native autoregressive-latency ratio. The resource
ledger now includes failed attempts,unique chip intervals and bounds on
interrupted scientific work. The CPU suite passes169 tests.
[Pilot progress](results/parcae_low_depth_pilot_20261024/REPORT.md) ·
[Inference counts](results/parcae_low_depth_pilot_20261024/inference-work.json) ·
[Observed costs](results/parcae_low_depth_pilot_20261024/resource-ledger.json).

Four of12 rate trials are independently scored. At0.002,ordinary scores
3.102339,one-pass Parcae3.256649 and recurrence3.297224; recurrence is0.040575
worse than one-pass here. Ordinary improves to2.994108 at0.004. These are
single tuning-seed results,not the selected recipes or main comparison.
All16 chips continue the remaining opportunities unchanged.

The one reserved full-gradient follow-up has CPU-only work tables prepared:
identical sampled forward counts,with differentiated tails changed from1 toN.
At the same1.858147e18 matrix-work budget its endpoint would be2.909B original
targets versus3.334B for truncation. This is unlaunched preparation only; its
conditional advancement and separate hardware/profile gates remain in force.
[Prepared cost](results/parcae_full_gradient_preparation_20261025/report.json).

**The first two independent301M TUNE scores are complete.** At the lowest
registered rate0.002,ordinary scores3.102339 nats and true one-pass Parcae
scores3.256649. Ten opportunities remain; no arm-level conclusion or rate
selection has been made. All16 chips continue with the recurrent trial and
higher rates. The controls and full screen stay frozen.
[Current scores](results/parcae_low_depth_pilot_20261024/tuning-v2/scores-progress.json).

**All16 chips are back in the frozen scientific LR screen.** The runtime-input
qualification now passes all three inventories,independent first updates,
trained and recovered complete-global gradients,replica equality and exact
state/RNG loads.100 varying-schedule updates measure491,376 ordinary,420,279
true one-pass and399,894 recurrent real targets/second per eight-chip job.
Overall trained-gradient errors are0.0213%,0.1085%,0.0773%,well below2%.
Both original20-update scientific prefixes also have identical complete
model/optimizer digests after the execution change. This establishes a
performance repair,not a prediction-quality gain.
[Recovery report](results/parcae_runtime_scalars_20261024/recovery-v2/report.json) ·
[Frozen new screen](results/parcae_low_depth_pilot_20261024/tuning-v2/plan.json) ·
[Prefix equivalence](results/parcae_low_depth_pilot_20261024/tuning-v2/runtime-prefix-equivalence.json).

The fixed follow-on queue now waits for complete predecessor receipts. It
will independently validate selected301M gradients,then register and execute
the finite one-seed matched-work/equal-data pilot. Main training has not started.
The zero-update gradient-capture driver passes independent CPU checks for both
qualification models. [Queued stages](results/low_depth_pilot_queue_20261024/recovery-v2/plan.json).

The natural-code target set remains149 original identifiers. Before scientific
scoring,the full prefix is fixed at1008 tokens and the random-state key uses
only the absolute next source index,independent of document hashes,the target's length or
content. Short views preserve original RoPE positions and hidden-state slices.
The numerical intervention rehearsal passes exactly at both supported exits.
Its initial failure identified the pinned upstream optional attention-mask
argument being ignored; an explicit mask adapter is used only in the numerical
fixture,while scientific forwards use unchanged upstream causal attention.
[Scoring-view policy](docs/NATURAL_CODE_CONTEXT_EVALUATION_20261024.md).

The CPU suite passes168 tests. No new held-out quality result or terminal
significance has been reported; the research goal remains active.

**The scalar-input fix restores measured scheduled throughput, but qualification
needs one retained-state recovery.** Ordinary and recurrent100-update intervals
with distinct LR/decay values reach491,376 and399,894 targets/second per eight
chips. Both uninterrupted/replayed native paths complete with exact state loads.
A separate gradient-check controller claimed the chips during an inter-phase
gap, stopping the third qualification inventory and interrupting CPU audits.
All four completed native paths are retained; the next execution will run only
the missing inventory and CPU audits, with explicit predecessor phase closure.
The follow-on pilot queue correctly stopped on that failed prerequisite. No
main fit or held-out score has run.

The next-global-gradient capture/checker qualification is currently using the
instance, with zero updates. Syntax-only selection also froze149 authentic
code identifier targets across the three declared distance bands; no model
scores were used. [Natural-code panel](results/natural_code_context_20261024/report.json).

**The first LR screen is stopped for runtime infeasibility before any held-out
scoring.** Changing schedule scalars cost about4.4seconds/update versus1.1–1.2
at constant values. The installed native `fill_.Scalar` registration specializes
on the value; the projected cooldown would exceed the frozen phase/screen caps.
No cap, source freeze or scientific outcome was changed. Both20-update states,
partial trajectories and951.152seconds of screen execution are preserved.
[Stop evidence](results/parcae_low_depth_pilot_20261024/tuning/runtime-budget-stop.json).

The runtime-tensor alternative gives bit-identical complete CPU model/optimizer
state across changing-scalar updates. All16 chips are now qualifying it with
the three actual inventories,100 varying-schedule warm updates,independent
gradients/first updates and full-state recovery. It changes scalar ingestion,
not optimizer mathematics. Only a separate new screen may adopt it after these
gates pass. The four rates,source exposure,initial weights,selection policy and
3hour budget stay fixed. [Performance policy](docs/PARCAE_RUNTIME_SCALARS_20261024.md)
· [Current qualification](results/parcae_runtime_scalars_20261024/plan.json)
· [Prospective amendment](docs/LOW_DEPTH_TUNING_RUNTIME_AMENDMENT_20261024.md).

**The stopped small-core scientific learning-rate screen**
It compares recipe-matched ordinary, true one-pass Parcae and sampled low-depth
Parcae, each with four independent rate opportunities at approximately301M
natural targets. Fresh seed11 weights are distinct from qualification weights;
each endpoint finishes its own cooldown. Pinned upstream CPU FP32 scores on
consumed TUNE select each arm's rate. No new main fit, fresh TEST, terminal alpha
or scientific success is implied. The whole screen is capped at3hours, including
CPU scoring. [Frozen screen](results/parcae_low_depth_pilot_20261024/tuning/plan.json)
· [Protocol](docs/LOW_DEPTH_TUNING_PROTOCOL_20261024.md).

**All10 complete model/training/profile cases pass**, including100 warm updates,
independent full-global trained gradients, prescribed actual updates,32-replica
equality and fresh-process full-state recovery. The common512-window batch is
faster than the smaller qualified batch. Measured FP16 rates per eight-chip job
are479,674 ordinary,435,366 one-pass and394,766 recurrent real targets/second.
The chosen execution uses FP16 body/FP32 head and masters, explicit8192 unscale,
eager FP32-polar Muon and fused unscaled AdamW. Compiled/eager BF16-polar failures
remain recorded; this is an explicit numerical deviation from the release.
[Complete engine report](results/parcae_low_depth_qualification_20261024/attempt4/REPORT.md).

The finite scientific-driver integration also passes: two arms each execute
20+20+10 updates, recover exact complete model/optimizer/RNG state, retain source
and work counters, and finish independent cooldown branches. Those100 updates
are qualification only. The first driver attempt failed before updates because
of a communication-port collision; explicit distinct non-ephemeral phase ports
resolved the observed failure without SDK, driver or host-network changes.
[Driver report](results/low_depth_driver_qualification_20261024/attempt2/REPORT.md).

Fresh seed1231 main initializations, fixed sampled-depth tables and integer work
endpoints are prepared but have not trained. At the common1.858147e18 estimated
matrix-work budget, ordinary receives3.646B real targets, true one-pass3.597B,
and sampled recurrence3.334B. Unlike old head-only estimates, this ledger charges
all32 replicas' Muon matrix work as well as neural forward/backward work. Pointwise
operations, communication and elapsed chip time are separate. Main launch still
requires selected-rate numerical checks and a finite execution registration.
[Prepared endpoints](results/parcae_low_depth_pilot_20261024/preparation-plan.json).

The CPU suite passes162 tests. The research goal remains active: no
compute-efficiency advantage has been demonstrated.

**The common-FP32 depth/connection campaign is complete and independently audited.**
All endpoints finish their own decay, share seed419/original weights/natural
source order, and pass full-state copy/readback. Maximum native/CPU score error
is7.25e-7; all47 frozen scientific sources remain unchanged.

| Model | Targets | DEV NLL |
| --- | ---: | ---: |
| Ordinary | 1.005B | 2.606833 |
| Cheap connection | 1.005B | 2.574477 |
| Large-core fixed recurrence | 1.005B | 2.578227 |
| Ordinary, matched estimated work | 1.700B | 2.497956 |
| Cheap connection, matched estimated work | 1.700B | 2.477058 |

The repeated core improves ordinary at equal data by0.028606 nats but loses to
cheap connection by0.003749. At matched estimated matmuls it loses by0.080271
and0.101168 respectively. The cheap connection remains the stronger control.
One consumed-DEV pair, no training-seed significance, fresh TEST or terminal
alpha. This closes the expensive-core recipe at this budget.
[Completed report](results/depth_connection_20261024/REPORT.md) ·
[Exact summary](results/depth_connection_20261024/summary.json) ·
[Receipts](results/depth_connection_20261024/execution-costs.json).

**Next experiment: a small2-block core at low depth, with strong one-pass
controls.** CPU preparation passes for recipe-matched ordinary, true one-pass
Parcae and sampled lambda2/minimum1/last1-gradient Parcae. All shared backbone
weights are cloned; one-pass and recurrent family have identical complete
initial weights and141.96M parameters. Expected executed depth is2.135335.
The earlier eight-pass CPU preparation is preserved but never launched natively.
The new charter anchors each main arm to the old ordinary1.700B complete work
budget, charges replicated Muon matrix work, reserves one controlled full-gradient
follow-up and specifies an authentic-code distant-context diagnostic. Final
throughput, endpoint/tuning allowances and a scientific launch remain pending.
[Scientific charter](docs/LOW_DEPTH_SCIENTIFIC_PILOT_20261024.md) ·
[Prepared engine](results/parcae_low_depth_qualification_20261024/REPORT.md).

**The spare-chip optimizer check found a real qualification failure before
scientific use.** Three prescribed actual-shape updates agree across32 replicas,
with AdamW error below6.14e-6, but Muon updates differ from independent CPU by
10.26–10.44%, above the frozen2% family gate. The unstarted optimizer, FP16 and
training-engine queues are suspended with zero native phases; all source freezes
are preserved. The separate FP32 zero-update gradient queue subsequently completed and passed.

A same-version CPU/native eager/compiled numerical trace narrows the discrepancy
to BF16 polar arithmetic; eager reduces the error but still misses the signed
third-step gate at2.32%. No gate is relaxed. The explicit FP32-polar variant now
passes independent actual-shape CPU updates: maximum1.65e-6 step-relative error,
4.72e-8 moment error,152.374s after imports, zero chips/research updates. The
pinned reference changes exactly one dtype; the candidate is separate code.
All future arms would share the qualified policy and report its cost/deviation.

**All16 chips are now in the new qualification pipeline.** The paired native
FP32 complete256-window gradients pass with relative errors4.18e-7 ordinary
and5.19e-7 recurrent, identical replicas and unchanged weights;151.612s active,
zero optimizer updates. The explicit FP32-polar optimizer arithmetic and recovery now also pass for
both models. Maximum primary Muon update error is0.13855%; exact state loads
and complete replicas agree. Recovery-only resumed updates equal uninterrupted
ones in this fixture, with no implication for complete future trajectories.

The initial replay was stopped by the unchanged host reserve before it was
qualified. The correction retained both passing primary captures, replayed only
the third update, reduced owned workload caps to128GiB each/256GiB aggregate,
and passed in92.135s. Every failure and original source is preserved. No gate,
SDK or scientific arithmetic was weakened.

The FP16-gradient and complete low-depth engine stages subsequently passed;
the scientific screen above now uses the qualified common geometry. Model,
initialization, source/noise/depth tables and numerical thresholds remain fixed.
Earlier BF16-dependent and stopped admission revisions remain preserved.
No scientific fit launches automatically from a software qualification.
[Native gradient pass](results/parcae_native_global_20261024/native-report.json) ·
[Optimizer/recovery pass](results/parcae_optimizer_precise_20261024/REPORT.md) ·
[Current engine](results/parcae_low_depth_qualification_20261024/attempt3/REPORT.md).

**The paired 3B FP32 continuation is complete and independently audited.** At
3,014,637,826 real targets, ordinary DEV NLL is **2.405907** and fixed two-pass
NLL is **2.369851**: a **0.036056-nat equal-data gain**. Both models have
150,436,608 parameters and completed the registered learning-rate decay.

| Domain | Ordinary NLL | Fixed NLL | Gain |
| --- | ---: | ---: | ---: |
| Web | 2.879228 | 2.838200 | +0.041028 |
| Math | 2.091221 | 2.056372 | +0.034849 |
| Code | 1.300630 | 1.278284 | +0.022345 |

This is one reused paired seed103 on consumed DEV, with BF16 history below
2.010B and qualified FP32 thereafter. It is a descriptive longer learning
curve, not fresh significance, a pure timing effect or an efficiency win.
All six whole-DEV independent FP32 scores and full optimizer/model copy audits
pass. Maximum native/CPU NLL difference is 0.000000808; all 37 frozen scientific
source hashes remain unchanged. Copies are on this instance.

[Completed report](results/mature_fp32_20261024/REPORT.md) ·
[Exact summary](results/mature_fp32_20261024/summary.json) ·
[Standalone learning curve](results/mature_fp32_20261024/figures/paired-learning-curve.png)
([SVG](results/mature_fp32_20261024/figures/paired-learning-curve.svg),
[PDF](results/mature_fp32_20261024/figures/paired-learning-curve.pdf)).

## Completed follow-ups and precision qualification

The **5.100B ordinary compute control is complete and independently audited**:
ordinary2.342369879 versus fixed3B2.369851103. Ordinary wins by0.027481223 nats.
Both endpoints completed decay; total estimated training matmuls match within
0.003051%. Differing precision-history fractions are not matched chip-time costs.
This one-pair consumed-DEV efficiency result is negative.
[Complete resource comparison](results/mature_fp32_compute_20261024/REPORT.md).

The **trained inexpensive-connection pilot is complete**. At1.005B targets,
ordinary NLL is2.606832753 versus connection2.574477293, a0.032355460-nat gain.
Parameters, original weights, source order and estimated matmuls match. Both
train in FP32 from random initialization, each with its own four-LR301M screen.
All selected TUNE and all six main whole-DEV/state-copy audits pass. This is one
predeclared seed419; the gain is descriptive, not seed significance or a latent-
depth success. It supplies a stronger architectural baseline for recurrence.
[Complete connection report](results/reinjection_pilot_20261024/REPORT.md).

The review correctly identified shared-container failure cleanup as an execution
risk. Both scientific campaigns finished without that interruption. Historical
frozen sources remain unchanged. New launches use separate ownership-checked
containers: ordinary failures stop their own workers; shared host-memory/disk
emergencies can stop managed workers globally. A real CPU-worker test verified
that cancellation preserves the other container and its existing worker.
[Isolation proof](results/native_execution_isolation_20261024/report.json).

The zero-update native FP16 full-gradient qualification **passed both complete
256-window mature objectives**: ordinary gradient relativeL2 error0.423%, fixed
1.173%, against the pinned CPU FP32 references. Maximum NLL discrepancy is
0.00002049; replica gradients agree and weights stay unchanged. This took182s
using all16 chips in separate containers, an FP32 vocabulary head, FP32 master
gradients and manual loss scale8192. Host drivers/XLA and SDK are unchanged.
No optimizer, recovery, throughput or learning adoption is implied. The active
scientific campaign deliberately retains its qualified FP32 path.
[Qualification protocol](docs/FP16_NATIVE_GRADIENT_PROTOCOL_20261024.md) ·
[Completed evidence](results/fp16_native_gradient_20261024/report.json).

The CPU feasibility screen passed0.7251% scaled-gradient error; unscaled FP16
failed at8.86%. All active scientific comparisons kept their frozen FP32 path.
A retrospective whole-document oracle gives only0.000651 nats improvement for
mixture depth selection and still loses to matched-work ordinary in every seed.
That deprioritizes document-level routing for those existing checkpoints; it
says nothing universal about token routing or new architectures.
[CPU precision report](results/cpu_fp16_precision_screen_20261024/REPORT.md) ·
[Oracle report](results/document_depth_oracle_20261024/REPORT.md).

The research goal stays active. Fresh confirmation panels remain unscored, no
terminal alpha is spent, and no latent-computation efficiency win is claimed.
The active scientific comparison retains the inexpensive connection as a
strong control and distinguishes its gain from the value of another core pass.

## The established three-seed result

The earlier 1B experiment and sealed TEST analysis remain unchanged:

| Model | Targets | Relative estimated matmuls | Mean TEST NLL |
| --- | ---: | ---: | ---: |
| Ordinary | 1.005B | 1.000 | 2.587896 |
| Fixed two passes | 1.005B | 1.692 | 2.535153 |
| Variable-depth mixture at two passes | 1.005B | 1.692 | 2.575693 |
| Ordinary, matched estimated work | 1.700B | 1.692 | 2.483895 |

Fixed's equal-data gain is0.052743 nats, with nominal paired-seed95% interval
[0.041749,0.063737], positive in all three seeds. Its compute-matched deficit is
0.051258. Variable depth loses to fixed at depth two; supported-depth dependence
does not establish monotonic refinement or learned halting. The original TEST
panel is consumed. [Frozen TEST report](results/variable_depth_20261023/test/REPORT.md).

## Qualified fresh data and preserved numerical findings

The new natural-source confirmation reserve is complete: 1,697 documents and
2,261,216 targets across TUNE/DEV/TEST. All14,700,735 prior accepted IDs were
screened. Independent raw-row/full-token reconstruction, coverage, exact-ID and
source-group checks pass. No model has scored these panels. The auditor did not
independently rerun every approximate minhash comparison; that limitation remains
explicit. The memory repair reuses the exact frozen candidate pool and leaves
both stopped predecessors and costs preserved.

[Fresh-reserve report](results/fresh_confirmation_reserve_20261024/v3/REPORT.md).

The earlier BF16 mature run stays stopped at its unchanged numerical guards.
The full-model gradient controls distinguish BF16 error from the native fast
FP32 SDPA precision gap. Disabling that attention route reduces the fixed
full-global FP32 gradient error from0.029312 to0.00012068. The public checkpoint-free
reproducer and upstream contribution notes are preserved; nothing was sent
upstream. The failed strict future-trajectory bit test remains failed; the
prospective numerical policy promises exact state on load and qualified
arithmetic, not bit-identical future process-restart trajectories.

[Numerical diagnosis](results/trained_gradient_diagnosis_20261024/REPORT.md) ·
[Numerical recovery policy](docs/FP32_NUMERICAL_QUALIFICATION_20261024.md) ·
[OSS notes](docs/OSS_NOTES.md).

The historical fixed2B scoring audit stays failed at−0.000201372 nats against
its old BF16 readout. Its independent FP32 remeasurement supplies a common-precision
parent anchor without waiving that failure. An FP32-head/BF16-body CPU gradient
screen reduced error to3.37%, still above2%; no native adoption followed.
The first cheap-control CPU checker also failed: it packed reference Q/K/V
gradients incorrectly. Its repaired, independently checked fixture gives5.13e-7
relative gradient error and zero loss difference, with the original failure and
a regression test preserved. These are numerical diagnostics, not learning gains.

The 148-test CPU suite passes after the latest source changes; targeted
recovery/admission tests also pass after the final input-integrity guard. Natural sources
only; no synthetic research corpus, external teacher, RL or larger model.
All failed/stopped jobs, tuning, qualification, compilation, data preparation
and CPU tails retain separate receipts. Concurrent intervals must not be added
as sequential reservation time. Existing old results/checkpoints are preserved.

[Previous full status and historical links](docs/archive/STATUS_BEFORE_FP32_3B_COMPLETION_20261024.md) ·
[Ongoing research charter](docs/ONGOING_RESEARCH_CHARTER.md).

An unapplied local SDPA dtype-policy contribution draft now has seven CPU gate fixtures and clean patch-application checks. It was neither installed nor submitted upstream. Patched native/cache/performance validation remains pending. [Draft](contributions/sdpa_dtype_policy/README.md).

The independent Parcae/RINS source review and pinned release-code audit are
complete. CPU candidate injection/all-gradient, finite-difference and truncated-
tail checks pass. The review identifies baseline-depth and executed-work
accounting choices that must be resolved for a faithful reproduction. This is
preparation, not another trained result; the active trainer does not import the
candidate. [Research triage](docs/PARCAE_RINS_REPRODUCTION_TRIAGE_20261024.md) ·
[CPU audit](results/parcae_reference_readiness_20261024/REPORT.md).

The **whole Parcae architecture candidate now passes CPU reference checks**,
including actual143M/144M parameter widths at128 positions. All logits and
gradients match the pinned reference exactly in seven checked cases, with
causal masking and an independently counted, head-inclusive matmul ledger.
The final measured CPU attempt takes22.07s,3.47GiB peak RSS and zero chips or
optimizer updates. This is numerical preparation, not a learning result.
The CPU suite includes inventory, truncation, causality and executed-work
regressions. Native graphs, sampled-depth DDP, optimizer, recovery and throughput
remain qualification gates; the active trainer does not import this candidate.
[Port scope and evidence](docs/PARCAE_CPU_PORT_20261024.md).

**Parcae preparation now includes CPU optimizer/recovery, static-stage AOT and
independent complete256-window TRAIN gradient references.** The reference work
uses261,664 real targets per model and takes448.09s from launch to report, with
zero chips or learning updates. CPU optimizer update differences stay below
6.88e-6 relative to reference-step size, with exact Muon moments and complete
nested-state replay. A report-writer failure remains preserved. Native behavior,
sampled-depth DDP, actual optimizer learning and throughput are still unqualified.

The paired native gradient gate is registered and queued behind completion of
depth-connection-v1, with no chip reservation while waiting. It has a30-minute
cap, local/global2% gradient gates and latest start01:00 UTC October25. Revision2
fixes an unstarted scheduler's late-readiness cutoff ordering; the canceled
zero-chip plan/source and receipt are preserved. Its12 source hashes are frozen
separately from the47 scientific files. Do not change either active bundle.

The executed-work audit finds Muon matrix updates add21.69% to ordinary neural
matmuls at32 ranks/global256 windows;64 ranks at the same batch doubles that
overhead. This is matrix work, not a latency estimate. New budgets must include
optimizer matmuls and measured time; old reports retain their declared scopes.
[Preparation, receipts and accounting](docs/PARCAE_OPTIMIZER_AND_STAGES_20261024.md) ·
[Native queue status](results/parcae_native_global_20261024/native-status.json).

Actual-shape Parcae CPU optimizer vectors now pass for both models: three
prescribed updates against the independent released algorithm, maximum update
relative error1.65204e-6 and named-moment error4.72221e-8. This takes124.485s
with zero chips and zero research updates. An initial report-directory failure
is preserved. A separate bounded native optimizer/recovery queue was registered behind
the native-gradient pass; it cannot displace the scientific campaign. Revision2
corrects an unstarted capture's buffer/tied-alias inventory before any native
attempt. CPU vectors and optimizer algorithm are unchanged. Its update/moment
gates apply separately to Muon and AdamW. Fresh-process full-state load/replay,
replica equality and ownership-scoped cancellation are required.
[Exact preparation and canceled attempt](results/parcae_optimizer_global_20261024/REPORT.md) ·
[Optimizer hardware protocol](docs/PARCAE_NATIVE_OPTIMIZER_PROTOCOL_20261024.md).

The data-free inventory audit identifies73.728M lookup-only value-embedding
parameters, about52% of the Parcae inventories. Another24.576M form the tied
input/output table; its dense output projection still counts as matrix work.
A recurrence comparison therefore needs the same-parameter depth-one Parcae
family, alongside strong ordinary controls. Unique parameter matching alone
does not match this allocation of capacity or executed work. The meta inventory
uses the default positional capacity2048; native qualification remains1024.
[Unique inventory and optimizer families](results/parcae_work_accounting_20261024/parameter-inventory.json).

A separate Parcae FP16-autocast body/FP32-head CPU screen passes on eight
natural TRAIN windows:0.07089% ordinary and0.06823% recurrent gradient error,
with every named family below2%. Three CPU tests cover disabled-policy function
identity, master/head/gradient dtypes and static-stage AOT. Wall time143.435s,
zero chips/updates. No scientific model or running precision policy changed.

Its now-suspended zero-update native gradient queue was registered to wait for the preceding FP32-gradient
and optimizer/recovery gates to pass, then uses all16 chips with a30-minute
cap and latest start03:00 UTC October25. Waiting assigns no chips. Original
CPU references remain authoritative; full global/local gradient families,
replicas, weight identity and fallbacks are checked. This prospective route
does not yet qualify mixed-precision training, optimizer integration, mature
states, trainer recovery or100 warm updates. The47 scientific source hashes
and the12/16/28 separately frozen qualification bundles are unchanged.
[CPU FP16 screen](results/parcae_cpu_fp16_20261024/REPORT.md) ·
[Conditional native plan](results/parcae_native_fp16_20261024/REPORT.md).
