# Authored arithmetic does not confirm the broad prediction benefit

Both registered arithmetic-conclusion gates fail. The latent model is slightly
worse than the strongest ordinary control and slightly better than pauses,
with both conditional source-group intervals spanning zero. This is not a
demonstration of arithmetic transfer and does not change the broad DEV result.

| Predictor | Full answer NLL | Last-32-prefix NLL | All answer tokens correct |
|---|---:|---:|---:|
| latent | 2.116429 | 2.172279 | 2/134 |
| ordinary | 2.102801 | 2.180442 | 1/134 |
| pause | 2.134533 | 2.192005 | 2/134 |

The full-context latent advantage is **−0.013628 nats versus ordinary**,
conditional 95% interval **[−0.062810, +0.035827]**, and **+0.018103 versus
pauses**, **[−0.035153, +0.072263]**. The fixed useful-effect screen was
+0.03 with a positive interval against both. There are 403 answer tokens across
134 original authored conclusions from 114 source groups.

Full-context utility is +0.055849 for latent, +0.077640 for ordinary and
+0.057472 for pauses. The latent interval includes zero. More context dependence
would not itself establish a better predictor or an arithmetic algorithm.

The 110 exploratory intermediate expressions also do not distinguish a useful
latent gain: +0.043467 versus ordinary [−0.024729, +0.111999] and −0.020890 versus
pauses [−0.080419, +0.038388]. They are not promoted to a replacement primary.

On the nine code-name cases from five groups, latent is worse by 1.311103 nats
than ordinary and 0.652573 than pauses, with nominal conditional intervals
favoring both controls. Coverage is too small for a general code conclusion.
All three models benefit from the supplied earlier code context.

These are fixed seed-9901 models, selected previously on broad NLL, scored on
all 253 existing reused-DEV cases. Authentic source targets and supplied worked
steps remain unchanged; no teacher, generated labels, tuning, new fit or fresh
reserve was used. All-answer accuracy is teacher-forced argmax over the fixed
answer span, not unrestricted solving or termination.

Exact sequential CPU FP64 readout passes six independent full-prefix checks
(maximum log-probability error 1.42e−14). All parameters remain unchanged.
Execution took 432.87 seconds, peak RSS 1.23 GiB, zero chips. No larger fit or
terminal confirmation is promoted by this result.

[All scores and intervals](report.json) · [Frozen design](../../../docs/INTERLEAVED_ARITHMETIC_DEV_20261029.md).
