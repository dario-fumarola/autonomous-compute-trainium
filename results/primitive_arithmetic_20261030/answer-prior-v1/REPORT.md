# Constant TRAIN-answer priors calibrate the near-floor accuracy

A predictor that ignores the prefix and always returns the most frequent TRAIN
answer gets9/256 elementary cases and3/259 harder cases correct. Each panel uses
its corresponding elementary or multi-operation TRAIN frequency table. Choices
are written before opening the evaluation answer files; no DEV answer selects
the guess. Complete literal equality and answer-prefix equality coincide here.

For context, the already published equal-data models solve13/256 elementary
cases for both latent and ordinary, versus7/256 for pauses. On the harder panel,
latent solves5/259 and ordinary/pauses4/259. These small margins over a constant
guess do not establish a general calculation procedure. They do not invalidate
the continuous NLL gains or quantify training-seed uncertainty.

The TRAIN counts require both the complete premise and answer to fit one packed
window:9,536 elementary occurrences and4,386 multistep occurrences. These are
slightly below the earlier first-answer-window census because an answer crossing
the window boundary is now excluded. No result is deleted or reclassified.

This is a post-hoc development calibration, not a new confirmation benchmark.
No model inference, weights, fresh reserve, accelerator or optimizer update is
used. Runtime1.117 wall seconds,0.681GB peak RSS. No further fit is admitted.

[Frozen rule](../../../docs/AUTHORED_ANSWER_PRIOR_20261030.md) ·
[Every operator/bin and source bindings](report.json).
