# Mixed numerical-round training: complete positive development screen

All three128,025,288-target fits finish their complete cooldowns and independent mature gradient/full-state audits. The common Native readout passes its independent FP64 checks before scores are accepted. Every endpoint also passes the separately registered TRAIN sequential-consistency guard.

| LR | K2 broad NLL | SupportedK3 broad NLL |
|---|---:|---:|
|0.0003|3.723270|3.723079|
|0.001|3.263546|3.262119|
|0.003|3.625396|3.625132|

**LR0.001 is the prespecified K2-selected interior optimum.** Its NLL3.263546 improves0.113565 over the retained ordinary control and0.127582 over trained constant pauses. It also improves0.017762 over the prior fixedK2 model. All three domains favor this mixed model over the ordinary and pause controls.

| Model/readout | Web | Math | Code | Broad |
|---|---:|---:|---:|---:|
| Selected mixedK2 |3.763593|2.929225|2.097728|3.263546|
| Selected mixedK3 |3.761979|2.927876|2.096782|3.262119|
| Ordinary |3.869100|3.049119|2.229139|3.377111|
| Trained pauses |3.895149|3.053362|2.216832|3.391128|

This fixes the original sampled-TRAIN consistency failure while retaining the prediction lead. Full sequential deployment on all384 broad DEV windows is separately registered and active; the small TRAIN consistency diagnostic does not replace it. K2/K3 are numerical approximation rounds, not different numbers of generated thought tokens. There is one latent per source token.

The mixed schedule costs5.592856times ordinary nominal training matrix work at equal data. Both original and mixed screens use one seed and reused DEV, and this is an adaptive follow-up after observing the original approximation mismatch. The controls retain their independently selected learning rates. No fresh confirmation, general reasoning, allocation or training-efficiency claim follows.

Maximum mature gradient-family relative L2 error is6.23023911e-05, below the unchanged0.02 rule. Complete numerical checks and cost ledgers are retained. Passing the frozen gates permits a separately bounded conventional-control qualification; no1B main or confirmation is automatically admitted.

[Analysis](analysis.json) · [All numerical receipts](report.json) · [Execution costs](execution-costs.json) · [Consistency result](../mixed-convergence-v1/REPORT.md).
