# Low-depth pilot progress

All12 registered rate trials completed301,465,908 natural targets and their own cooldown. Independent pinned-upstream FP32 CPU scoring selected each arm separately on consumed TUNE. No main or fresh-confirmation scores are available yet.

| Peak LR | Ordinary | True one-pass | Sampled recurrence,exit2 |
| --- | ---: | ---: | ---: |
| 0.002 | 3.102339 | 3.256649 | 3.297224 |
| 0.004 | 2.994108 | 3.082854 | 3.105910 |
| 0.008 | 2.926390 | 2.993713 | 3.013839 |
| 0.016 | 2.884985 | 2.950986 | 2.957179 |

All three arms selected0.016,the upper boundary of the finite grid. Their optima are not bracketed. The candidate is0.006193 nats behind its parameter-exact one-pass control at this horizon; ordinary is0.072194 better than the candidate. These screen results do not determine the longer pilot. No extra rate is added after seeing them.

All three selected301M checkpoints pass independent full512-window gradient checks: overall errors0.223%,0.257%,0.264% for ordinary,one-pass and recurrence,with maximum loss discrepancy0.00000460 nats. Fresh-seed1231 main training now performs actual updates on all16 chips. The one-pass/recurrent first segments are active; ordinary waits for an eight-chip slot. Every main1B/endpoint is checked again. Qualification and tuning weights never initialize a main fit. [Gradient gate](../low-depth-gradient-selected-v1/report.json) · [Frozen main plan](main/plan.json).

Ordinary has140,781,888 parameters; both Parcae paths have141,963,840. Recurrence versus true one-pass is the exact parameter-control comparison. Keep the strongest eligible one-pass/cheap-connection challenger for later fresh confirmation.

[Observed tuning figure](figures/tuning.png) · [Complete screen and selection](tuning-v2/report.json) · [Screen costs](tuning-v2/execution-costs.json).

The runtime-scalar performance repair passes actual updates,full gradients,recovery and100 varying-schedule warm updates. Its first20 scientific model/optimizer digests match the earlier execution exactly. Earlier failures and interrupted work remain in the resource ledger.

Independent actual-size operator counts measure159.049GFLOPs for one family pass and195.691GFLOPs for two at1024 positions,including the50.332GFLOP tied head. That23.038% forward-work increase is not measured native cached-generation latency. [Inference work](inference-work.json).

A short read-only runtime sample during the final eight-chip fit observed85.0% active-core busy time. The other half was already released for CPU scoring,so whole-node busy time was42.5%. This four-interval observation is not tensor-peak utilization or a campaign average. [Tuning-tail sample](runtime-busy-sample.json). A later main-fit sample has all64 logical cores active and84.6% node busy time over11 accepted intervals. [Main runtime sample](main-runtime-busy-sample.json) · [Campaign ledger](resource-ledger.json).

A separate prospective CPU-only audit will repeat the completed main endpoints with latent randomness keyed solely by original window position. The frozen pooled evaluator uses a whole-document content-hash-derived seed; this is unavailable before prediction,although no decoding of that hash or effect on the ranking is established. Preserve the frozen scores and report the policy sensitivity separately. The natural-code context diagnostic already uses position-only randomness. No training recipe or fresh reserve changes.

The full-gradient follow-up remains conditional and unlaunched. No fresh TEST/reserve or terminal alpha has been used. The research goal remains unachieved.

A separate TRAIN-only numerical screen at the301M main checkpoint finds bit-identical forward outputs under last-one versus full gradients. The raw FP32 gradients differ by15.1% on the8 prespecified windows;this is the intentional gradient-rule difference,not a failed precision gate. FP16 body with8192 scaling and an FP32 head agrees with independent FP32 gradients to0.108% overall and0.111% in the worst named family under full backpropagation. Both numerical precision comparisons pass. The diagnostic took94.8 CPU seconds,17.907e12 estimated matrix FLOPs and zero accelerators/optimizer updates. It cannot establish native throughput,recovery,complete-batch fidelity or a learning benefit. [Full-gradient CPU evidence](../low_depth_full_gradient_cpu_20261025/report.json).
