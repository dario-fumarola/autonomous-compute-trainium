# Prospective final-quarter recurrence allocation

The fixed-two and current-prefix diagnostic families are closed at their original endpoints. Their failed advancement gates do not admit4B or a router. This is a new bounded schedule hypothesis at the existing2B fixed-two nominal-work budget,using the existing qualified model and ordinary next-token objective.

Hypothesis: learning the ordinary path first and paying for the second pass only in the final quarter can preserve enough useful recurrent learning to improve prediction at fixed total training work. The experiment tests when to buy the existing computation,not another architecture or auxiliary objective.

## Primary-source rationale and limits

[Elastic Latent Recurrence,section6.5](https://arxiv.org/html/2610.12582v1) reports that delaying mixed depths to the final quarter retains84% of its five-loop improvement with lower estimated cost. Its0.5B/300B-token gated setup and same-data comparisons differ from ours;this is no forecast of an ordinary-more-tokens win here.

[Chen et al.,sections4.2/A.2.3](https://arxiv.org/html/2609.19107) finds benefits from model growth,including conventional untied growth. Its tied experiment grows2→4 with additional boundary operators. Therefore a candidate that wins our static ordinary8 screen must face a separately tuned conventional-growth challenger before independent confirmation. General growth benefits cannot be attributed specifically to weight sharing.

[LoopMTP,appendixD](https://arxiv.org/html/2608.03624v1) offers a later objective hypothesis,but its three-loop training costs2.49 times its non-looped reference. It does not establish our matched-training-work claim. Ordered/cosine objectives are deferred;this experiment retains ordinary NTP.

## Frozen design

Keep the existing2/2/2,width768 recurrent family,original random initializers,real source order,tokenizer,precision,primitive FP32 attention,full gradients and runtime-scalar optimizer. Keep and charge the original zero-weight auxiliary head to preserve the qualified engine and direct fixed-two reference. It is not a scientific auxiliary objective. Deployment removes it as before.

For a registered endpoint ofU updates,run exactly one core pass through update floor(3U/4),then exactly two throughU. All32 ranks and16 microbatches use the same depth at a given update. Both applications receive gradients. The integer switch,source counts,realized token fraction and complete work vector are frozen before training. It is the final quarter of updates;the actual source-token fraction is reported rather than assumed exactly25%.

Use the original50M-target warmup and half-horizon cooldown,including cooldown before and after the switch. Do not reheat or reset LR,weights,moments,RNG or source order. Every endpoint finishes its own complete cooldown to zero. A fresh process at the switch must restore the exact complete state under the same plan. This is from-random-initialization training,not continuation from a cooled checkpoint.

TUNE uses seed4127,three rates0.008/0.016/0.032,and576 updates /the original301M targets. All three opportunities finish their cooldowns. Stage endpoints121/432/576 preserve the full history. Count the121-update prefixes inside each fit. Require full512-window independent FP32 gradient audits after the prefixes and at the selected final endpoint,with the original2% overall/family and0.003-nat gates. Select the minimum original TUNE NLL,lower LR breaking ties;an edge winner stays unbracketed. Equal-rate opportunities and all costs for the historical conventional tuning remain disclosed.

Measure100 actual warm one-pass updates from progress20→120,and100 actual two-pass updates from440→540 after the registered switch. Include actual changing runtime scalars,optimizer,data and logging;separate cold compilation and checkpoint overhead. No scientific MAIN before the complete TUNE numerical/timing/restore gates and a measured complete budget.

MAIN,if separately admitted,uses both reused development seeds5231/17293. The sole primary comparator is the completed independently tuned conventional ordinary8 at1,272,689,958,300,549,120 nominal matrix work. Compute the largest complete staged endpoint within that same budget. Include all model/head/attention/backward and replicated optimizer matrix work;report hardware time separately. Compare the exact original same-seed ordinary8 and fixed-two endpoints,not a favorable cross-seed pairing. No terminal positive is possible on these consumed data/seeds.

Both paired MAIN fits must finish regardless of an early unfavorable quality mean,subject to numerical/resource/deadline gates. Require exact prefix and switch restoration,all-rank RNG/replica checks,an independent full512 gradient audit on the first two-pass objective before learning after the switch,and final mature audits. Score only the final supported two-pass exit on the original consumed DEV curve panel with frozen60/20/20 weights. Unsupported deletion/extra passes do not establish thinking.

The development advancement criterion is a mean gain at least0.01 over ordinary8 with both seed points positive and every required audit passing. Report source-group and nominal seed uncertainty separately. A positive must additionally survive conventional model growth and equal-source-exposure ordinary controls before any fresh confirmation. The strongest eligible ordinary alternative remains the standard. Failure closes this75% timing recipe at this budget;no automatic timing/LR sweep,larger model,4B exposure or new objective follows.

## Admission and cost limits

Preparation is CPU-only,zero model updates and no held-out scoring. It validates byte-identical original initializers,accepted6.5B source/count/anchor prefixes,integer work floors and frozen depth/readback tables. Scientific admission is separate.

Prospective caps:TUNE7,200 seconds and a later paired MAIN14,400 seconds,with an original composite21,600-second execution cap beginning once at TUNE admission. TUNE admission cutoff23:30 UTC October26. Registration/recovery cannot renew the original composite deadline. Each native/CPU phase is separately capped,failures cancel only owned workers,and initiating exceptions survive cleanup. Use all16 chips for valid independent work.

Maintain512GiB host memory,35GiB filesystem and10,000 free-inode reserves,and reserve complete projected artifacts before admission. Storage maintenance may share byte-identical immutable closed numerical h0 arrays while preserving every path,byte hash and scientific checkpoint,and recording original per-path metadata;it needs separate verified receipts. Do not delete source/corpus/model/optimizer evidence to make room. Count preparation,tuning,audits,failed/stopped work and reserved-instance opportunity without overlapping double counts.

No fresh reserve,TEST,terminal alpha,router or claim of independent confirmation is admitted. At least five adequately powered fresh paired training seeds and genuinely fresh source groups remain required by the unchanged charter.
