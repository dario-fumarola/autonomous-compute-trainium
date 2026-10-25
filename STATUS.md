# Status — 2026-10-25, 09:31 UTC

**All twelve tuning scores and four selected gradient gates pass;the
fresh two-seed MAIN study is running on all16 chips.** Independent FP32 scores use the same
556-document consumed TUNE panel and 301,465,908 training targets per fit.

| Cell | Selected peak LR | TUNE NLL |
| --- | ---: | ---: |
| One-pass NTP | .032 | 2.938677 |
| One-pass future loss | .032 | 2.935304 |
| Recurrent NTP, two-pass readout | .032 | **2.917065** |
| Recurrent future loss, two-pass readout | .032 | 2.918299 |

Plain NTP recurrence gains 0.021612 nats over independently tuned one-pass NTP
and 0.018239 over one-pass future loss. Adding future loss instead worsens
recurrence by 0.001234;the factorial interaction is -0.004607. This is TUNE,
not a fresh-seed result. All selected rates are upper-edge winners and their
optima remain unbracketed. Recurrence uses 1.232× estimated training matrix
work at equal exposure;no matched-compute advantage is established.
[All twelve scores and contrasts](results/future_span_tune_20261025/primitive-v2/complete-grid-readout.json).

The original future-objective primary remains unchanged. A separate NTP
secondary discovery readout was registered after seeing its complete TUNE
grid and before MAIN admission. It reuses only the already declared final
readouts and requires both main seeds to beat both trained one-pass controls
and improve at a supported second inference pass. It cannot override the
original primary or serve as independent confirmation.
[Disclosed secondary protocol](docs/NTP_SECONDARY_DISCOVERY_20261025.md).

All four selected checkpoints pass independent complete512-window gradient
checks;the largest backbone discrepancy is0.330%,below the fixed2% gate.
The complete TUNE stage took1h51m,24.652 assigned native chip-hours and29.628
whole-instance opportunity chip-hours,including CPU tails;these costs are
separate and are not added.
[Completed tuning audit](results/future_span_tune_20261025/primitive-v2/complete-tuning-closure-audit.json).

The controller admitted all eight1.005B-target MAIN fits:four cells across
fresh seeds5231/7283. Both first seed5231 one-pass cells have already performed
20 actual optimizer updates and10,467,614 real targets each. All16 chips are
assigned. No held-out MAIN outcome is available. MAIN retains its original
8h cap,ending17:29 UTC;architecture,objective,rates,data and decision rules
remain frozen. No terminal confirmation has been opened.
[Actual fresh-update receipt](results/future_span_main_20261025/primitive-v2/native-start-receipt.json).

A separate pre-outcome CPU audit reconstructs the complete matrix ledger and
checks a possible one-pass work-matched continuation:2365 updates/1.238B
real targets,0.021527% below recurrence's1.005B matrix budget. Its own cooldown
would start at1182 updates,after the preserved stable960-update state. No
checkpoint was adopted,no scores read and no control run admitted;cross-plan
full-state recovery still needs qualification.
[Work/source feasibility](docs/FUTURE_SPAN_COMPUTE_READINESS_20261025.md).

All **218 CPU tests pass**. Resource guards pass. Fresh confirmation sources
and terminal alpha remain untouched;the overall research goal is active and
unachieved. Complete the primary factorial readout and the separately disclosed
NTP secondary before selecting a follow-up.

# Earlier status — 2026-10-25, 08:14 UTC

**All six one-pass 301.5M-target tuning endpoints are scored;the factorial
comparison is unfinished.** Both one-pass three-rate grids are complete. Independent FP32 scores use
the same 556-document TUNE panel with 759,473 real targets.

| Cell | Peak LR | Cooled TUNE NLL |
| --- | ---: | ---: |
| One-pass NTP | .008 | 2.993821 |
| One-pass NTP | .016 | 2.949029 |
| One-pass NTP | .032 | **2.938677** |
| One-pass future loss | .008 | 2.991968 |
| One-pass future loss | .016 | 2.950876 |
| One-pass future loss | .032 | **2.935304** |

Both best tested rates are the upper grid edge;their optima remain unbracketed.
Best tested future loss gains 0.003373 nats over best tested NTP,below the 0.01
useful-effect threshold. This is a one-seed TUNE observation,not a recurrent
interaction or confirmation. All twelve frozen opportunities and four selected-checkpoint gradient gates remain required.
No rate,horizon,weight,threshold or admitted budget changes.

All 16 chips are training the first two recurrent NTP trials. The main two-seed study is not admitted.
First scientific stable checkpoints pass independent full model/optimizer
readback and exact source/domain/anchor/work-counter checks. The first baseline
score aggregates and document pairing also reproduce independently.
[Live scores](results/future_span_tune_20261025/primitive-v2/scores-progress.json)
· [Checkpoint audit](results/future_span_tune_20261025/primitive-v2/startup-checkpoint-audit.json).

A separate 20.1-second CPU diagnostic of the already saved qualification packets
confirms the future loss changes the shared backbone gradient:its initial
contribution norm is 5.09% of NTP's,and the native contribution matches CPU to
0.0276% relative error. This uses identical initial weights,source,depths and
noise. It is an initial numerical-channel check,not a prediction result or
explanation of later learning. It changes no gate and uses zero new device work.
[Initial signal audit](results/future_span_initial_signal_20261025/report.json).

All **214 CPU tests pass**. The research goal remains active and unachieved;
confirmation sources,seeds and terminal alpha remain untouched.

# Earlier status — 2026-10-25, scientific tuning admitted

**The objective study is running real scientific updates on all16 chips.**
The complete four-cell hardware qualification and all four scientific-driver
integration audits pass. Continuation,replay,source/work counters and separately
cooled branches are independently checked. The discovery queue admitted the
12-case301M-target LR screen from fresh seed4127 weights. The first two trials
were observed at210 optimizer updates and109.9M real targets. No new held-out
scientific result is available yet.
[Actual startup receipt](results/future_span_tune_20261025/primitive-v2/native-start-receipt.json)
· [Driver pass](results/future_span_driver_20261025/primitive-v2/report.json).

The four cells are deterministic-one/sample-recurrence × NTP/future-presence.
All keep the same source order and initial parameter inventory. The future
span is a training target only;no teacher,generated corpus or future input
enters the predictor. Each cell has the same three LR opportunities and its
own completed cooldown. Four distinct selected-checkpoint gradient gates
precede the separate two-seed1.005B main study. Main remains conditional on
that complete pass;fresh confirmation and terminal alpha remain untouched.

Actual100-update measurements give a conservative3.28h TUNE projection under
a4h cap,and6.63h main projection under the prospective8h cap. Caps and the
pre-outcome amendment are public;they cannot renew after admission. The useful
effect threshold and all absolute/interaction controls remain unchanged.
All **214 CPU tests pass**. No positive scientific claim is established.

# Earlier status — 2026-10-25, 07:28 UTC

**The four-cell objective hardware qualification passes completely.** All four
paths perform121 actual optimizer updates plus a fresh-process replay update;
all12 independent objective,gradient and optimizer audits pass. Model,
optimizer and CPU RNG restore exactly,and all32 replica states agree.
The largest trained body-gradient relative error is0.113%,under the fixed2%
gate. These are qualification weights,not scientific results.
[Completed hardware evidence](results/future_span_qualification_20261025/math-sdpa-v2/REPORT.md).

Measured100-update throughput is476k real targets/second for one pass and326k
for recurrence per eight-chip job. Qualification costs3.885 assigned chip-hours
and9.233 whole-instance opportunity chip-hours,including the CPU-only audit
period. The two quantities are reported separately.

The final driver checks run on all16 chips. Both one-pass cells pass independent
continuation,replay and shorter-branch checks;the two recurrent cells are
running. Scientific fitting is not admitted until all four pass.
The waiting discovery queue is now registered against the correct primitive
execution and exact fresh initializers. It enforces all12 fixed TUNE
opportunities and four distinct selected checkpoint-gradient gates.
[Driver](results/future_span_driver_20261025/primitive-v2/status.json)
· [Waiting science queue](results/future_span_science_queue_20261025/primitive-v2/status.json).

The conservative measured projections are3.28h TUNE and6.63h main. Before any
new scientific scores,the main cap is prospectively amended6h→8h,retaining
both1.005B-target paired pilot fits;TUNE remains4h. No admitted deadline can
extend automatically. Architecture,objective,grid,source order,threshold and
confirmation policy are unchanged.
[Measured budgets](results/future_span_objective_budget_20261025/measured-qualified-budget.json).

A project-independent CPU reproducer and an upstream-review draft are ready.
The numerical diagnostic isolates the saved-output backward correction;
no SDK patch or upstream submission is made.
[Review draft](docs/OSS_SDPA_MATH2_REVIEW_DRAFT_20261025.md).
All **214 CPU tests pass**. The overall research goal remains unachieved;
completed negative results and sealed confirmation data remain intact.

# Earlier status — 2026-10-25, 06:50 UTC

**The small-core recipe is closed at its declared budget;the next hypothesis
changes the predictive objective.** The completed trained controls show no
useful recurrent advantage. The prefix-only-noise audit is now complete:
maximum pooled difference0.00001191 nats,and the negative primary comparison
is unchanged. No new TEST,confirmation seed or alpha was used.
[Closed pilot](results/parcae_low_depth_pilot_20261024/main-recovery-v2/REPORT.md)
· [Research reset](docs/OBJECTIVE_RESET_20261025.md).

The explicit attention-primitives fallback passes all eight compiled numerical
fixtures on all32 replicas,plus complete512-window gradients at both mature
checkpoints with exactly matched source/depth/noise/full-state inputs. It fixes
the saturated operator fixture:query relative error48.4%→0.0361%. Full-model
aggregate changes are small:recurrence1.763%→1.690%,one-pass0.454%→0.459%.
This does not explain away the completed scientific results. The cast-only
workaround remains failed;its costs and the earlier packaging failure remain.
[Qualified numerical path](results/sdpa_math2_investigation_20261025/native-v3/report.json).

All16 chips are now assigned to the finite four-cell objective update/recovery
qualification. The two one-pass cells each completed121 actual optimizer
updates,exact fresh-process full-state/RNG recovery and100 warm updates at
approximately476,000 real targets/second per eight-chip job. Recurrent cells
and independent complete-objective/optimizer audits remain in progress. These
are qualification weights,not scientific fits or new held-out results.
[Current qualification](results/future_span_qualification_20261025/math-sdpa-v2/status.json).

A separately registered finite scientific-driver check waits for the COMPLETE
hardware pass. It cannot launch TUNE/main or consume confirmation. The planned
objective comparison remains one-pass/recurrent × NTP/future-presence,with
fixed horizons/weights and explicit stopping rules. Strong ordinary+future,
compute controls and independent confirmation remain required for an eventual
positive claim. **211 CPU tests pass**.

# Earlier status — 2026-10-25, 06:36 UTC

**The small-core pilot is complete and does not support scaling this recipe.**
All nine fits/complete-gradient gates and all declared FP32 scoring readouts
finished. At matched work,true one-pass wins2.448818 versus recurrence2.450674;
at equal1.005B data,ordinary2.615775 versus recurrence2.616335. One seed,
consumed DEV,no new confirmation or useful0.01-nat recurrent gain.
[Completed result](results/parcae_low_depth_pilot_20261024/main-recovery-v2/REPORT.md).

The trained one-pass control on149 authentic code targets is also complete;
neither matched-work nor equal-data intervals support a recurrent advantage.
The original joint context-use criterion remains inconclusive.
[Exploratory trained control](results/natural_code_context_20261024/onepass-control-v1/report.json).
The prefix-only-noise sensitivity audit is running on CPU.

The compiled attention check reproduces a numerical discrepancy on all32
replicas. On the32-position saturated FP16 fixture,the installed path has48.4%
query-gradient relative error;the CPU-promising cast-only workaround still has
37.2% and **fails** the unchanged2% gate. Forward relative error is0.0316%.
No optimizer update or scientific launch occurred. The preceding snapshot
packaging failure and both attempt costs are preserved. A bounded alternative
using differentiable FP32 attention primitives is being qualified next;the
installed SDK is untouched. Future-objective science remains stopped.
[Compiled failure](results/sdpa_math2_investigation_20261025/native-v2/operator-audit.json).

The next scientific proposal changes the objective,not model size:a fixed
2×2 one-pass/recurrent × NTP/future-span comparison. The original natural
source,teacher-free constraints and separate one-pass future-objective control
remain. Only a complete numerical/update/recovery/100-warm-step pass can admit
it. This is a prospective hypothesis,not a positive result or a paper replication.

# Earlier status — 2026-10-25, 06:12 UTC

The separately trained one-pass control is now scored on the same149 authentic
code targets. At matched work,candidate depth2 gains0.02563 nats,but the paired
repository interval is[−0.06716,+0.11644]. At equal data,it loses0.04920,with
interval[−0.14067,+0.04687]. Neither supports a trained-control advantage.
These exploratory controls were added after the candidate readout,with the
same targets,positions,prefix-only noise and band weights. No new dataset or
confirmation claim is introduced. This107.7second CPU readout made596 forwards,
zero updates and used zero native chips. All **204 CPU tests pass**.
[Trained control](results/natural_code_context_20261024/onepass-control-v1/report.json).

The attention qualification queue remains live,waiting for ordinary's final
branch/gate. Original mature source/depth/noise/checkpoint fingerprints are now
registered for exact pairing before any new native outputs. A numerical
improvement will not be attributed to the workaround unless those inputs match.
[Pairing registration](results/sdpa_math2_investigation_20261025/paired-model-input-registration.json).

## Earlier status — 06:02 UTC

All three cooled matched-work scores are available:ordinary **2.454545**,
true one-pass **2.448818**,recurrence at its declared two-pass exit **2.450674**.
The recurrent gain versus ordinary is0.003871,and it loses0.001856 to its
parameter-exact one-pass control. Neither comparison shows a0.01-nat useful
gain in this one-seed pilot. These are consumed-DEV source intervals,not
training-seed confirmation;the prefix-only-noise audit remains pending.
[Complete primary readout](results/parcae_low_depth_pilot_20261024/main-recovery-v2/three-arm-matched-work.json).

Ordinary's final equal-data branch is running. The compiled-attention numerical
queue is confirmed live and waits for the remaining native work/gates;all
future-objective queues remain stopped before hardware admission.

## Earlier status — 05:52 UTC

The authentic-code diagnostic is complete after an execution-only path repair.
At the matched-work checkpoint,extra supported depth improves full-context
identifier NLL by0.03277,but the full-versus-short interaction interval crosses
zero:[−0.00290,+0.07083]. The declared joint context-use criterion is therefore
**inconclusive**,not positive. Both1B mechanism intervals also cross zero.
The149 original targets/views,checkpoint and absolute deadline were unchanged.
[Completed readouts](results/natural_code_context_20261024/readouts/recovery-v3/report.json).

The earlier scorer looked for numerical receipts in the stopped predecessor's
directory. The recovery now takes the exact gate file explicitly;the failed
attempt remains recorded. All **203 CPU tests pass**. These CPU readouts do not
alter the running fits or the pending compiled-attention investigation.

The recurrent1B endpoint gains only0.000871 nats over its trained one-pass
control(2.616335 versus2.617206). Its own one-to-two supported-depth benefit is
0.014194,and0.017189 at the longer endpoint. These show depth dependence
within one fitted model,not a useful gain over separately trained one-pass
prediction. The primary two-pass exit is unchanged;three-pass readouts remain
secondary. Ordinary and the prefix-only-noise sensitivity audit are pending.

## Earlier status — 05:44 UTC

**The waiting future-objective queues are stopped before hardware admission.**
A CPU reproduction of the installed Math2 attention formulas finds large
query/key gradient differences on valid saturated FP16 software fixtures despite
close forward outputs. The installed SDK and running pilot remain unchanged.
A separately registered numerical queue will check actual compiled operators
on32 ranks and complete512-window gradients at the two mature checkpoints.
It waits for the current pilot's native work to finish and cannot launch training.
[CPU evidence](results/sdpa_math2_investigation_20261025/cpu-v1/report.json)
· [Finite hardware plan](results/sdpa_math2_investigation_20261025/native-v1/plan.json).

For the32-position fixture,installed-formula forward error is0.0316%,while
query/key gradient errors are67.1%/61.4%. Explicit FP32 attention inputs retain
the internal output through backward and reduce those CPU errors to0.0441%/
0.0414%. This is an operator fixture,not a language-model result or a verified
compiled-device defect. It does not establish why a scientific recipe lost.
No upstream submission is made. All **202 CPU tests pass**,including unchanged
parameters/RNG and CPU gradient equivalence for the local model wrapper.

The first cooled matched-work pair does not show a useful recurrence gain:
one-pass2.448818 versus recurrence2.450674 DEV NLL. Candidate gain is−0.001856;
the paired source-group interval is[−0.004313,+0.000240],conditional on seed1231.
Parameters match exactly,and nominal matrix budgets differ by0.00406%.
This does not provide training-seed significance. Ordinary and the remaining
equal-data readouts are pending;fresh confirmation stays sealed.
[Original primary contrast](results/parcae_low_depth_pilot_20261024/main-recovery-v2/first-matched-work-pair.json).

Recurrence's mature complete-gradient audit passes at1.763% overall,1.811% in
the largest named family,under the unchanged2% gate. One-pass's cooled1.005B
score is2.617206. The new operator probe investigates a numerical concern;
these actual fitted-model scores and prior negative results remain intact.

The separate CPU FP16 vocabulary-projection screen also passes at both mature
checkpoints:0.199%/0.244% gradient error,zero optimizer updates/chips. It does
not establish native speed or fidelity and is not applied to any training.
Its preceding metadata failure and unknown elapsed duration remain recorded.
[Projection screen](results/parcae_fp16_head_cpu_20261025/v2/report.json).

## Earlier status — 05:00 UTC

The separate future-objective discovery queue is armed and waiting,with zero
chips. It can register TUNE only after complete independent hardware and driver
passes and a measured conservative budget fitting the four-hour cap. Main then
requires all twelve frozen LR opportunities,four selected trained-gradient
passes and its own measured six-hour cap. It keeps four cells,two fresh paired
pilot seeds,completed cooldowns and all endpoint audits. It cannot launch fresh
confirmation or claim terminal success. All **199 CPU tests pass**.
[Registered waiting queue](results/future_span_science_queue_20261025/plan.json)
· [Exact execution rules](docs/FUTURE_SPAN_OBJECTIVE_PROPOSAL_20261025.md).

These are finite scientific caps after numerical qualification,not a prediction
that either full allocation will be consumed. A non-fitting budget or failed
prerequisite stops admission. Candidate advancement needs the objective
interaction,absolute gains and supported depth benefit in both pilot seeds;
stronger one-pass+future and compute controls remain outstanding obligations.
No objective TUNE or main fit is running yet.

## Earlier status — 04:47 UTC

The first cooled matched-work endpoint is independently scored: true one-pass
reaches **2.448818 DEV NLL at3.5966B natural targets**. Its complete512-window
gradient audit passes at0.454% overall relative error. Recurrence has finished
its3.3339B-target fit,with its endpoint audit/readout pending;ordinary and the
three separate cooled1.005B branches are unfinished. These are seed1231 results
on consumed DEV. No new recurrence advantage or efficiency result is established.
[Actual scores](results/parcae_low_depth_pilot_20261024/main-recovery-v2/scores-progress.json)
· [Numerical gates](results/parcae_low_depth_pilot_20261024/main-recovery-v2/gradient-progress.json).

The stable1B ordinary score is2.769278,versus2.773757 for true one-pass and
2.780522 for recurrence. Ordinary's first1B full-gradient check also passes
(0.0882%). A within-family control is therefore insufficient for a practical
positive claim:the strongest qualified ordinary/cheap-connection baseline must
receive the new objective too before any terminal comparison.

Fresh random initializations for the prospective2×2 objective study are prepared
at seeds4127/5231/7283,with exact state readbacks and no qualification or trained
weights reused. Source/anchor/full-gradient work tables cover continuation
without wrapping. This60.1second CPU preparation made zero updates,used zero
chips and opened no held-out panel. No scientific execution plan is admitted.
[Fresh preparation](results/future_span_scientific_preparation_20261025/report.json).

The scientific trainer,independent head-free evaluator and bounded execution
helpers now pass **192 CPU tests**. A separate watcher is registered to run a
finite driver check only after complete hardware qualification:actual updates,
exact optimizer/RNG loads,one-update replay,and an independently annealed
shorter branch in all four cells. It currently uses zero chips. Neither watcher
can automatically launch scientific tuning/main training.
[Driver queue](results/future_span_driver_queue_20261025/plan.json)
· [Prospective objective](docs/FUTURE_SPAN_OBJECTIVE_PROPOSAL_20261025.md).

The proposed learning-rate grid is prospectively centered on the preceding
family's0.016 boundary winner:0.008/0.016/0.032,three equal opportunities for
every cell. This amendment precedes all future-objective TUNE/main outcomes.
The new readout keys latent noise only by original prefix-window position;
future content,document identity and suffix length cannot change it. Hardware
gradients,actual recovery and measured100-update rates remain launch gates.
Fresh confirmation stays sealed and no terminal alpha is spent.

## Earlier status — 03:39 UTC

The finite four-cell hardware qualification is registered,but not admitted or
running. It must see all nine current-pilot fits and all nine numerical gates
with successful native closures before claiming chips;an idle inter-phase gap
is insufficient. Its separate90minute clock begins only on admission and
persists across controller resume. Passing it cannot launch a scientific fit.
[Frozen qualification plan](results/future_span_qualification_20261025/native-v1/plan.json)
· [Required checks](docs/FUTURE_SPAN_NATIVE_QUALIFICATION_20261025.md).

An actual-size CPU rehearsal also passes at depths1/2:the new training and
independent audit adapters agree to about1e-7 relative body-gradient error on
one original TRAIN window. This44.7second check has zero updates/native work.
[Adapter evidence](results/future_span_qualification_20261025/cpu-adapter-rehearsal/report.json).

All185 CPU tests pass,including exact named moment restoration for the new
head and a test that rejects incomplete predecessor receipts. Native actual
updates,full gradients,checkpoint recovery and100-step rates remain unverified.
The current pilot still occupies the instance;ordinary's completed1B checkpoint
awaits its queued numerical audit while one-pass and recurrence continue.

CPU preparation for the prospective objective experiment is complete. The
unchanged1.005B-target stream has60,105,879 complete future anchors out of
61,931,520 executed slots. Their1,863,282,249 auxiliary label presentations
reuse original source tokens;they are not extra unique corpus exposure. Global
anchor-prefix counts and the common qualification-only initializer restore
exactly. Full-gradient depth tables preserve the original exogenous forward
draws. This40.0second preparation used zero chips/updates and no held-out data.
[Prepared manifest](results/future_span_qualification_20261025/preparation-v1/manifest.json).

The sparse presence loss matches dense BCE values/gradients,including duplicate
labels and extreme logits. It avoids a dense vocabulary target/scatter without
changing the target or weight. The fixed-shape training adapter has separate
global token/anchor denominators passed as runtime tensors. All183 CPU tests
pass. Native full-batch gradients,updates,recovery and throughput remain gates;
no new native or scientific objective run has launched.

The proposed objective pivot now has a CPU-tested implementation: a separate
training-only head predicts token presence in the next32-position span,excluding
the immediate next token and any document boundary. The2×2 compares one-pass
and recurrence,each with/without that same objective. A positive interaction
must accompany absolute gains; equally helping both architectures would not
establish a recurrence benefit. [Bounded prospective design](docs/FUTURE_SPAN_OBJECTIVE_PROPOSAL_20261025.md).

The eight-window natural-TRAIN screen passes against the pinned independent
FP32 backbone and a separately implemented presence loss. Overall body gradient
errors are0.120%/0.108%;auxiliary-head errors0.0206%/0.0217%. Maximum combined
loss discrepancy is0.0000213. This105.6second CPU screen has zero updates,
17.325e12 matrix operations and no held-out access. It does not qualify native
training or establish a quality gain. All179 CPU tests pass,including causal
prefix invariance,exact zero-weight NTP gradients and removable-head equivalence.
No objective training has launched. [Numerical evidence](results/future_span_cpu_20261025/report.json).

**The unchanged small-core pilot is running on all16 chips after an execution
failure.** The original controller constructed an uppercase `stable1B` worker
identity,which the launcher's lowercase-only guard rejected. Its paired-stop
canceled ordinary after70 logged updates; there was no OOM. Both completed
one-pass/recurrent1B checkpoints survived. The original failure and costs remain
visible. [Failed execution](results/parcae_low_depth_pilot_20261024/main/report.json).

The recovery copies preserve every scientific tensor,moment,RNG state,cursor,
schedule and work counter exactly. Only the execution-plan signature changes;
disk readback verifies all non-signature state. One-pass and recurrence resume
their retained histories; ordinary replays the same original initialization and
source order because its interrupted prefix had no checkpoint. The original
09:21 UTC deadline remains fixed. New owned identities avoid the rejected label.
[Exact adoption](results/parcae_low_depth_pilot_20261024/main-recovery-v2/adoption.json)
· [Recovery plan](results/parcae_low_depth_pilot_20261024/main-recovery-v2/plan.json).

Independent first1B complete512-window gradient gates pass: one-pass0.176%
and recurrence0.113% relative L2 error. Both new native fits are making actual
updates. CPU context/noise watchers carry their original preregistrations and
deadline into explicit recovery executions; their stopped predecessors remain.
[Gradient evidence](results/low-depth-recovery-v2-gradient-gates/report.json).

The first independent main learning-curve scores are available:

| Stable-phase natural targets | True one-pass DEV NLL | Recurrence,exit2 DEV NLL |
| --- | ---: | ---: |
| 301,465,908 | 3.032445 | 3.045218 |
| 1,004,887,590 | 2.773757 | 2.780522 |

Recurrence is0.006765 nats behind at stable1B. These are seed1231 curves on
consumed DEV,not cooled endpoints,seed significance or an efficiency result.
Ordinary and the completed-decay equal-data/matched-work endpoints remain
pending. No new positive is established. [Actual scores](results/parcae_low_depth_pilot_20261024/main-recovery-v2/scores-progress.json)
· [Observed curves](results/parcae_low_depth_pilot_20261024/figures/learning-curves.png).

The owner proposed a prospective objective study: one-pass/recurrence crossed
with ordinary prediction/future-span supervision. Primary literature is being
checked while the frozen pilot runs. No new objective or gradient-rule training
has launched. A useful objective interaction and absolute prediction gains are
different requirements; improvements shared equally by both models would not
establish an additional-computation benefit. Prior efficiency negatives stand.

The CPU suite passes175 tests after recovery helpers,watcher restoration and
interrupted-work accounting.
Fresh confirmation remains unscored and no terminal alpha is spent. The overall
research objective remains unachieved.

## Earlier progress, retained chronologically

### 01:48 UTC

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

A TRAIN-only full-gradient precision screen passes at the immutable301M main
checkpoint. On the first8 next TRAIN windows,full/truncated forwards are
bit-identical; restoring earlier paths changes the raw FP32 gradient by15.1%.
That is intentional truncation bias,not numerical error. FP16 full-gradient
error is0.108% overall,0.111% at the largest named family. This94.8second CPU
subbatch used zero chips/updates and no held-out data. It establishes neither
native qualification nor a learning gain; the follow-up remains conditional.
[Exact numerical screen](results/low_depth_full_gradient_cpu_20261025/report.json).
The CPU suite still passes172 tests after adding the screen.

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
