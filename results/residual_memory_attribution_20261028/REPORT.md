# Correct memory explains little of the current prediction gain

The frozen diagnostic completes all448 broad-text views and339 source-verified dependency cases,with no updates. Authentic-memory,raw and local predictions reproduce. Neither primary dependency family passes the registered content-sensitive mechanism gate. The pretrained correction remains a small development result;its interpretation as useful source-specific rereading is weakened.

| Broad-text condition | Full32 NLL | Gain over frozen C0 |
| --- | ---: | ---: |
| Frozen C0 | 2.282112 | +0.000000 |
| Trained local nonlinear control | 2.277691 | +0.004422 |
| One-step,authentic memory | 2.275685 | +0.006427 |
| One-step,independent memory1 | 2.276416 | +0.005697 |
| One-step,independent memory2 | 2.276083 | +0.006029 |
| One-step,current slot only | 2.279499 | +0.002613 |

Replacing earlier memory removes only0.000731nats with donor1 (conditional95%[0.000125,0.001374]) and0.000398 with donor2 ([−0.000273,0.001071]). About89% and94% of the original gain survive,as descriptive ratios of point estimates. Current-slot-only attention loses0.003814nats versus authentic memory,but this changes attention normalization and available values. It is not evidence by itself that relevant earlier facts caused the gain.

The original current-token representation stays fixed and already contains ordinary-backbone context. This test concerns the value of accessing earlier representations again;it does not erase all past information from the predictor. Both independent donors match domain and have different source groups from the recipient and each other. K/V are recomputed separately per query,retaining only that query’s original current slot.

## Authentic dependency cases

| Family | Cohort0:whole-answer gain | Cohort1:whole-answer gain | Registered gate |
| --- | ---: | ---: | --- |
| code_binding | +0.020932 | +0.020963 | Fails |
| math_conclusion | +0.022678 | +0.015866 | Fails |
| math_intermediate_exploratory | +0.017880 | +0.019907 | Fails |

All whole-answer gain intervals include zero. No primary family establishes0.03nats gain and useful authentic memory relative to both donors in both source cohorts. Code bindings are lexical dependencies;their longer contexts also extrapolate beyond the branch’s training length. One longest code case was excluded before scoring because two distinct compatible donor groups were unavailable. All134 arithmetic conclusions and110 exploratory intermediates remain.

A positive secondary result is preserved:arithmetic first-target gains are0.035058 ([0.001183,0.072757]) and0.030271 ([0.001534,0.066881]). But first-target donor effects are near zero,the local control remains competitive,and the registered whole-answer/content gates fail. This cannot be promoted into a demonstration of arithmetic reasoning. Exploratory intermediate effects do not consistently replicate across cohorts.

## What the TRAIN-only inspection adds

After the outcome,two separately registered inspections use only the original512 TRAIN feature views,no targets or vocabulary projection. Attention is learned and substantially sharper:largest weight averages0.3657 versus0.0141 initially;entropy is57.5% of uniform versus99.2%. Replacing weighted attention by a uniform read changes the correction vector by21.3% on average,without measuring a new prediction score. The simple near-uniform-averaging explanation is unsupported.

A positional check finds2.28% attention mass on the first key and6.83% on the first four. The largest weight is in the first four on18.3% of queries. There is an early-position bias,but the attention is not predominantly a first-window-position pattern. Memory values share a large mean component (68.2% mean-energy ratio);whether that helps explain donor insensitivity remains a hypothesis. Sharp attention or changed states alone do not establish useful computation over relevant content.

## Limits,checks and next decision

Source-group intervals condition on the saved weights,reused recipients and chosen donor pool. They do not capture training-seed uncertainty or fully account for donor reuse across recipients;they are not fresh confirmation. Donor replacement is a distribution shift,and the arithmetic panel contains supplied source-written work. No new corpus,teacher label,answer-aware donor choice,router or unsupported depth is introduced.

All11 focused CPU tests pass. Maximum authentic-branch output discrepancy is3.815e-6;maximum saved broad NLL discrepancy1.431e-6;maximum historical independent C0 dependency NLL discrepancy1.050e-5,below the unchanged0.0002 gate. All model versions and scientific checkpoint hashes remain unchanged. The memory diagnostic takes330.04seconds on CPU;the TRAIN geometry/position checks take4.37/4.48seconds. No additional accelerator work.[Costs](complete-costs.json).

The experiment also reserves320 unused workspace source groups (192web,64math,64code) without scoring them. This is an accepted source-identifier reserve,not a reconstructed/tokenized panel. Source reconstruction,tokenization,deduplication and a frozen replication protocol are still required. Existing TEST and the older sealed confirmation reserve remain unopened;upstream pretraining exposure is unknown.[Reserve](source-reserve.json).

The two-step model is not promoted. The next defensible learning control would compare authentic-memory training with an otherwise identical branch trained using independent memory from the outset,before interpreting the attention architecture’s gain as content-dependent work. That control is not launched by this report. The current results support a transferable predictor correction;they do not yet establish reusable intermediate reasoning,autonomous allocation,matched-compute efficiency or the random-initialization hypothesis.

[Frozen protocol](../../docs/RESIDUAL_MEMORY_ATTRIBUTION_20261028.md) · [All scores,domains,cohorts and intervals](report.json) · [TRAIN geometry](../residual_attention_telemetry_20261028/report.json) · [TRAIN positions](../residual_attention_position_20261028/report.json)
