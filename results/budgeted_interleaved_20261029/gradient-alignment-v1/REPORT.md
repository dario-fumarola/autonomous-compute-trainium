# Gradient alignment across supported budgets

All three completed endpoints are included. These statistics use the existing
full-global TRAIN gradient captures at the same model weights and source batch
within each LR trial. No extra forward/backward pass, optimizer update or
held-out read was needed. Dot products and squared norms accumulate in FP64.

| LR | Zero vs one token | Zero vs two tokens | One vs two tokens | One-token K2 vs K3 | Two-token K4 vs K5 |
|---|---:|---:|---:|---:|---:|
|0.0003|0.837987|0.704928|0.944068|0.997950|0.998273|
|0.001|0.812408|0.718224|0.945076|0.993902|0.998376|
|0.003|0.883778|0.879806|0.966224|0.999571|0.999847|

Values are whole-model gradient cosines. Body-only and separate embedding/
readout results, plus norm ratios, remain in every endpoint receipt. Actual
budget comparisons use their registered numerical graphs; they are not exact
sequential-gradient comparisons.

At these three endpoints and this batch, the aggregate directions are aligned,
not opposed. The numbers therefore do not support a simple explanation in
which the supported budgets broadly cancel each other's gradients. They do
not rule out local interference, different optima or inadequate exposure.
AdamW preconditioning, weight decay and the entire training trajectory are
not represented by an instantaneous NLL-gradient cosine. Strong alignment does
not itself establish useful intermediate computation or good prediction.

This descriptive analysis was declared after the first fit's audits and before
the common DEV readout. It changes no scientific gate or selection. An initial
read-only summarizer used the wrong name for the output-weight tensor; an
empty-family guard stopped it before output. The corrected grouping uses the
actual `prediction_weight` name; no gradients or scientific states changed.

[Declared analysis](plan.json) · [Low LR](budget-lr0.json) ·
[Middle LR](budget-lr1.json) · [High LR](budget-lr2.json) ·
[Retained preparation error](preparation-error.json).
