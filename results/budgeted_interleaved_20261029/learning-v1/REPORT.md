# Supported zero/one/two latent training: all four development gates fail

All three learning-rate fits finish their1957updates and complete cooldowns.
Every endpoint passes all five mature gradient/state audits and both TRAIN-only
sequential-consistency screens. The interior LR0.001is selected by the frozen
two-token readout. Exact cached sequential evaluation confirms the negative.

| Same checkpoint's trained latent budget | Exact development NLL |
|---|---:|
| Zero | 3.487978 |
| One | 3.379654 |
| Two | 3.381733 |

The second token's gain is−0.002079nats, conditional95%source interval
[−0.003520,−0.000656]. Its apparent code gain0.001369has an interval spanning
zero; web and math point estimates favor one token. This is not useful scaling.

Two-token prediction loses0.118999nats to independently trained fixed-one and
0.093952to strongest tied ordinary. The zero-token exit loses0.110867to the
independently trained same-shape ordinary model. Thus the large within-model
one-over-zero gap does not establish a competitive computational benefit.
All four frozen advancement gates fail; no extension, router, confirmation or
prepared ephemeral fit is admitted by this result.

The experiment uses128,025,288authentic targets, one reused initialization and
the repeatedly inspected384-source development panel with60/20/20weights.
Intervals condition on these trained endpoints; they are not seed uncertainty.
Both fresh evaluation reserves remain unscored and terminal alpha remains zero.

Complete Native attempts cost16.234592assigned chip-hours, including all
three fits and common scoring. Whole-instance opportunity cost is
29.314901chip-hours, including CPU
audit/readout intervals; these are different ledgers and must not be added.
The selected fit uses4.454times strongest ordinary's nominal matrix work.
An independent reconstruction of the saved domain loss sums matches every
reported score. No failed numerical gate or score was discarded.

A design limitation is that changing global thought budget also changes all
previous source memories. These modes do not isolate an extra operation on an
unchanged lower-stage history. This motivates a distinct depth-causal software
reference, not a reinterpretation of this failed learning comparison.

[Complete comparisons](compare.json) · [All LR scores](select.json) ·
[Exact readout](readout.json) · [Execution costs](execution-costs.json) ·
[Independent aggregate check](independent-aggregate-check.json) ·
[Frozen protocol](../../../docs/SUPPORTED_LATENT_BUDGET_LEARNING_20261029.md).
