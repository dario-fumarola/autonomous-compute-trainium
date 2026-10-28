# Supported iteration does not beat the independently trained shallow model

Both new branches complete all 2,048 updates and pass mature numerical and recovery checks. Each receives the same 8,192 source views, 262,144 sampled natural targets, parameters, initialization and balanced schedule: 1,024 updates at each supported depth. The written-memory variant lets the second step read first-step states at earlier positions. The static control preserves the original earlier states. Both use an updated current-token slot.

| Predictor | One supported step | Two supported steps |
| --- | ---: | ---: |
| Static earlier memory | 2.276423 | 2.275580 |
| Updated earlier memory | 2.276430 | 2.275799 |
| Independently trained one-step control | 2.275685 | — |

Frozen C0 scores 2.282112; the local nonlinear control scores 2.277691. Suppressing earlier writes at the written model's second read, while still computing those writes and preserving the updated current slot, scores **2.275403**. This registered intervention is reported as an intervention, not a selected new trained winner.

The written model improves from its own shallow exit by **0.000631 nats**, conditional 95% interval **[0.000112, 0.001208]**. However, its shallow exit is 0.000745 worse than the independently trained shallow model. The second step recovers less than that deficit: its advantage over the independent one-step control is **−0.000113**, interval **[−0.000719, 0.000494]**. The static model's supported second-step gain is similarly small, 0.000843.

The newly computed earlier states do not supply the hoped-for benefit. Written-two trails static-two by 0.000219 nats. Its own earlier-write utility is **−0.000395**, interval **[−0.000632, −0.000161]**: suppressing those writes helps on this panel. Intervention distribution shift remains a limitation; the separate trained static control also supplies no useful written-memory advantage.

The original practical bars all remain visible. Written-two improves raw prediction by 0.006314, below 0.01; it does not beat the independent shallow model, its own supported shallow exit, or static-two by the required 0.005. The earlier-write intervention also has the wrong sign. **The useful-iteration gate fails.** A small supported-exit gain alone would have given a misleading impression of progress against the stronger shallow control.

All domains are retained in [report.json](report.json). Both variants still worsen mathematical-text NLL relative to C0. This is one branch seed and 116 repeatedly inspected evaluation source groups; conditional intervals do not measure training-seed variation. No fresh reserved group, new dependency case or confirmation data was scored. The result does not establish a universal negative about latent reasoning or pretraining from random initialization.

Sixteen focused tests pass, covering causality, zero-output identity, first-exit equivalence, all parameter gradients, query independence and controlled suppression of earlier writes. Full-size FP32 versus an explicit FP64 reference passes both supported exits in web, math and code at the unchanged numerical gates. Maximum mature named-gradient relative error is 3.899e-6. Complete model/AdamW/counter/query-RNG restoration and disposable next-update replay are exact. Previous raw/one-step scores reproduce with zero measured discrepancy; scientific checkpoint and source hashes remain unchanged.

Static and written fits take 353.48 and 367.46 CPU wall-seconds concurrently. The complete controller takes 449.92 seconds, including mature audits and evaluation. Training matrix work is respectively 2.180718e14 and 2.186079e14 operations, a 0.246% difference, including the vocabulary projection and backward path. No new Native or backbone updates occur. Shared feature/base/comparator costs and reserved-instance opportunity remain in the [cost ledger](complete-costs.json). This is an equal-source/update mechanism screen, not a full-work ordinary-continuation comparison.

Under the [registered stopping rule](../../docs/RESIDUAL_WRITTEN_MEMORY_20261028.md), this frozen-feature two-step mechanism is closed. No additional steps, coefficient search, larger backbone or confirmation follows automatically. Together with the [trained independent-memory control](../residual_independent_memory_20261028/REPORT.md), it narrows the current evidence to a useful small predictor correction: substantial value from matched additional context, newly written states and iteration remains unestablished. The broader thinking-during-pretraining goal is still open.
