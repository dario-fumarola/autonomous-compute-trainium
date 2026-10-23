# Exploratory cross-panel reference

Motivated during the variable-depth LR screen, before main fits: score all six
canonical old1B ordinary/shared endpoints (seeds23/47/71, their trained depths)
on the complete fresh DEV curve panel. Use the independent FP32 CPU model and
preserve every real target. Do not score these references on new sealed TEST.
No old or new checkpoint, exit, LR or primary decision is selected by this
readout. The original negative and its consumed TEST result stay unchanged.

This asks whether the old ordinary/shared gap persists on the new source-group
panel. Compare its three paired old-seed differences with the new recipe's
separately fitted three-seed DEV result when available. The two recipes use
different seeds and are not paired across recipes. A changed gap would not
identify beta2 as the cause: optimizer groups, warmup, LR selection horizon,
DDP topology and within-update routing also differ. This is a development
diagnostic, not another confirmatory contrast or a rescue of the old result.

The CPU-only readout can overlap native training. It is bounded to finish
before23:00 UTC and enters the CPU resource ledger. No model updates,
architecture change, synthetic source or external teacher are used.
