# Earlier solver gradients contribute materially

The no-update diagnostic completes at the saved billion-target checkpoint.
All three gradient rules produce **identical forward logits** on all six frozen
TRAIN windows at both K2 and K3. Their training gradients differ substantially.

| Retained backward path | K2 body cosine / relative error | K3 body cosine / relative error | Analytic matrix-work reduction |
|---|---:|---:|---:|
| Last refinement + final readout | 0.9606 / 28.11% | 0.9586 / 28.82% | 30.31% |
| Final readout only | 0.8228 / 57.66% | 0.8245 / 57.50% | 45.95% |

The reductions average the two original solver counts equally and keep every
forward operation. They are theoretical matrix counts, not measured accelerator
savings. The corresponding training work remains about 2.636 and 2.044 times
the strongest tested ordinary model per authentic target, versus 3.782 for the
full-gradient candidate. Exact finite campaign schedules have their own ledger.

Overall cosine is higher (about 0.978 for last-refinement credit and 0.909 for
readout-only credit), partly because the vocabulary-head gradient is identical.
That aggregate hides the larger change in the body and input embeddings. The
input-embedding relative errors are 41.6–43.9% and 83.5–84.4%, respectively.
Readout-only differentiation removes direct credit to the generated input;
the shared body still learns through its readout application. These are different
learning rules, not equivalent implementations of the current objective.

The diagnostic does not establish that either shortened rule trains badly or
well. It rules out calling early backward work negligible from these measurements.
No optimizer state changes and no new fit is admitted. The unchanged full-gradient
prediction results and active work-matched ordinary comparison remain primary.

Six focused tests pass, including complete original-gradient agreement,
independent frozen-prefix suffix gradients, finite differences, and work counts.
All model gradients are finite. The six per-window gradients are aggregated
with the original 60/20/20 domain weights; this is not a Native global training
batch or an AdamW trajectory comparison. No DEV or reserve scores are read.

Runtime is 26.135 wall seconds, 384.379 process CPU seconds, 4.148 GB peak RSS,
zero assigned chips and zero optimizer updates. Host work overlaps the ordinary
fit and is disclosed separately without double-counting reservation time.

[All gradient families and work counts](report.json) · [Frozen plan](plan.json) ·
[Protocol](../../../docs/LATENT_CREDIT_DIAGNOSTIC_20261030.md).
