# Completed low-depth pilot: no useful matched-work advantage

The frozen one-seed pilot and its declared endpoint readouts are complete.
The following summary supersedes the historical progress snapshots below.

| Arm | Natural targets | Parameters | Cooled DEV NLL |
| --- | ---: | ---: | ---: |
| Recipe-matched ordinary | 3.646B | 140,781,888 | 2.454545 |
| True one-pass family control | 3.597B | 141,963,840 | **2.448818** |
| Sampled recurrence,declared exit two | 3.334B | 141,963,840 | 2.450674 |

Complete training matrix work is approximately1.858e18 per arm. Recurrence's
matched-work gain is−0.001856 nats against its parameter-exact one-pass
control and+0.003871 against recipe-matched ordinary. Neither clears0.01.
At the separately cooled equal-data1.005B endpoints,the corresponding gains
are+0.000871 and−0.000561. These are one fitted seed on consumed DEV,not
training-seed significance. The original last-one backward rule and independently
selected.016 rates remain part of this completed recipe.
[Complete main report](main-recovery-v2/report.json)
· [Three-arm matched-work comparison](main-recovery-v2/three-arm-matched-work.json).

The prefix-only latent-noise sensitivity audit is also complete. Its maximum
whole-panel score change is0.00001191 nats. The primary matched-work gain over
one-pass becomes−0.001854,and the equal-data gain becomes+0.000873;the
interpretation is unchanged. This supplementary audit preserves the original
content-keyed frozen readout separately. It does not validate future-dependent
noise as deployable. New studies use prefix-available randomness.
[Completed audit](../low_depth_prefix_noise_audit_20261025/recovery-v2/report.json).

The supported second pass benefits the same recurrent checkpoint,but that alone
does not establish superiority to a trained one-pass alternative. This pilot
provides no useful efficiency win. The later future-span factorial study is a
separate development experiment with different rates,full gradients,attention
arithmetic and prefix-only evaluation;cross-study scores cannot isolate any
one change. No fresh confirmation or terminal alpha was used in this pilot.

# Historical progress snapshots

# Low-depth pilot progress

The first cooled primary matched-matrix comparison is available:one-pass
**2.448818** at3.5966B natural targets versus recurrence **2.450674** at3.3339B.
Candidate gain is−0.001856 nats,with conditional paired source-group interval
[−0.004313,+0.000240]. Both models have141,963,840 parameters. These are one
paired seed on consumed DEV;the interval is not training-seed significance.
The ordinary endpoint and remaining equal-data readouts are pending.
[Exact partial comparison](main-recovery-v2/first-matched-work-pair.json).

The original frozen fits continue. Separately,an installed attention arithmetic
discrepancy has been reproduced on CPU numerical fixtures. A new zero-update
hardware qualification will test compiled behavior and a local explicit-FP32
attention call;it does not modify the SDK,these fits or their conclusions.
[Operator investigation](../sdpa_math2_investigation_20261025/cpu-v1/report.json).

The main pilot has independent stable-phase DEV scores; completed-decay endpoints remain pending. At301M targets,true one-pass scores3.032445 versus recurrence3.045218; at1.005B they score2.773757 versus2.780522. Recurrence is0.006765 worse at stable1B. These are one fresh training seed1231 on consumed DEV,not a cooled comparison or a positive result. [Current scores](main-recovery-v2/scores-progress.json) · [Observed curves](figures/learning-curves.png).

The original controller failed when its uppercase `stable1B` execution identity was rejected by the lowercase worker guard. Both completed1B checkpoints survived; the ordinary prefix was canceled after70 logged updates with no checkpoint. The new recovery execution adopts all scientific state bit-exactly,changes only execution signatures/identities and preserves the original09:21 UTC deadline. Ordinary replays the same initialization/data. All interrupted work stays in costs; no model/loss/schedule or numerical threshold changes. [Failure](main/report.json) · [Exact adoption](main-recovery-v2/adoption.json) · [Recovery plan](main-recovery-v2/plan.json).

All12 registered rate trials completed301,465,908 natural targets and their own cooldown. Independent pinned-upstream FP32 CPU scoring selected each arm separately on consumed TUNE. No fresh-confirmation scores are available.

| Peak LR | Ordinary | True one-pass | Sampled recurrence,exit2 |
| --- | ---: | ---: | ---: |
| 0.002 | 3.102339 | 3.256649 | 3.297224 |
| 0.004 | 2.994108 | 3.082854 | 3.105910 |
| 0.008 | 2.926390 | 2.993713 | 3.013839 |
| 0.016 | 2.884985 | 2.950986 | 2.957179 |

All three arms selected0.016,the upper boundary of the finite grid. Their optima are not bracketed. The candidate is0.006193 nats behind its parameter-exact one-pass control at this horizon; ordinary is0.072194 better than the candidate. These screen results do not determine the longer pilot. No extra rate is added after seeing them.

All three selected301M checkpoints pass independent full512-window gradient checks: overall errors0.223%,0.257%,0.264% for ordinary,one-pass and recurrence,with maximum loss discrepancy0.00000460 nats. First1B one-pass/recurrent gradients independently pass0.176%/0.113%. Fresh-seed1231 main training performs actual updates on all16 chips;ordinary first1B and one-pass continuation are active. Every main1B/endpoint is checked again. Qualification and tuning weights never initialize a main fit. [Selected-gradient gate](../low-depth-gradient-selected-v1/report.json) · [First1B gates](../low-depth-recovery-v2-gradient-gates/report.json).

Ordinary has140,781,888 parameters; both Parcae paths have141,963,840. Recurrence versus true one-pass is the exact parameter-control comparison. Keep the strongest eligible one-pass/cheap-connection challenger for later fresh confirmation.

[Observed tuning figure](figures/tuning.png) · [Complete screen and selection](tuning-v2/report.json) · [Screen costs](tuning-v2/execution-costs.json).

The runtime-scalar performance repair passes actual updates,full gradients,recovery and100 varying-schedule warm updates. Its first20 scientific model/optimizer digests match the earlier execution exactly. Earlier failures and interrupted work remain in the resource ledger.

Independent actual-size operator counts measure159.049GFLOPs for one family pass and195.691GFLOPs for two at1024 positions,including the50.332GFLOP tied head. That23.038% forward-work increase is not measured native cached-generation latency. [Inference work](inference-work.json).

A short read-only runtime sample during the final eight-chip fit observed85.0% active-core busy time. The other half was already released for CPU scoring,so whole-node busy time was42.5%. This four-interval observation is not tensor-peak utilization or a campaign average. [Tuning-tail sample](runtime-busy-sample.json). A later main-fit sample has all64 logical cores active and84.6% node busy time over11 accepted intervals. [Main runtime sample](main-runtime-busy-sample.json) · [Campaign ledger](resource-ledger.json).

A separate prospective CPU-only audit will repeat the completed main endpoints with latent randomness keyed solely by original window position. The frozen pooled evaluator uses a whole-document content-hash-derived seed; this is unavailable before prediction,although no decoding of that hash or effect on the ranking is established. Preserve the frozen scores and report the policy sensitivity separately. The natural-code context diagnostic already uses position-only randomness. No training recipe or fresh reserve changes.

The full-gradient follow-up remains conditional and unlaunched. A teacher-free future-span objective study is being investigated prospectively;it has not launched or changed this pilot. No fresh TEST/reserve or terminal alpha has been used. The research goal remains unachieved.

A separate TRAIN-only numerical screen at the301M main checkpoint finds bit-identical forward outputs under last-one versus full gradients. The raw FP32 gradients differ by15.1% on the8 prespecified windows;this is the intentional gradient-rule difference,not a failed precision gate. FP16 body with8192 scaling and an FP32 head agrees with independent FP32 gradients to0.108% overall and0.111% in the worst named family under full backpropagation. Both numerical precision comparisons pass. The diagnostic took94.8 CPU seconds,17.907e12 estimated matrix FLOPs and zero accelerators/optimizer updates. It cannot establish native throughput,recovery,complete-batch fidelity or a learning benefit. [Full-gradient CPU evidence](../low_depth_full_gradient_cpu_20261025/report.json).

Algebra from those same raw norms gives a full/tail gradient cosine of0.991912 (7.29degrees),with the tail norm90.96% of the full norm. Thus the15.1% gradient change is largely a norm difference,not evidence of opposing aggregate updates. This adds no model forwards or held-out access. [Derived geometry](../low_depth_full_gradient_cpu_20261025/derived-gradient-geometry.json).
