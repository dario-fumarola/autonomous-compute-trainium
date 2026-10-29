# Supported-budget gradients are aligned on the audited TRAIN batch

This descriptive analysis reuses the existing mature gradient captures at all three learning-rate endpoints. It performs no new model passes, updates or held-out scoring.

| Learning rate | Zero/one cosine | One/two cosine | Zero/two cosine |
| --- | ---: | ---: | ---: |
| 0.0003 | 0.915333 | 0.954893 | 0.798864 |
| 0.001 | 0.885142 | 0.967600 | 0.793048 |
| 0.003 | 0.874717 | 0.964039 | 0.872145 |

All body, embedding and readout-family dot products are positive too. This does not support a broad opposite-gradient explanation at these particular endpoints. It does not establish that shared-budget training is harmless: these are raw gradients on one batch, not AdamW updates, a trajectory study or a causal diagnosis.

The analysis takes 3.44 CPU seconds and consumes no accelerator work. A first invocation stopped before admission because the last audit receipt was not yet present; no captures or model were loaded. It was repeated after all audits passed. No training rule, learning-rate selection or advancement gate changes.

[Plan](plan.json) · [All families and norm ratios](report.json) · [Pre-admission stop](../gradient-preparation-v1.json).
