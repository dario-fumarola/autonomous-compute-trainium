# TRAIN-only gradient compatibility

This zero-update diagnostic uses two reserved TRAIN windows, no held-out
examples. At initialization, first/final gradient cosine is0.993934. After103
disposable joint-objective updates it is0.936756. These measurements do not
support gradient cancellation as the cause of early learning difficulty.

At the mature final-only checkpoint, hypothetical shallow and deep objective
gradients have cosine0.170376. The shallow-gradient norm is8.8859 versus2.0241
for the deep objective. Its intermediate readout was never supervised as an
exit, so its larger TRAIN loss is not a supported test-time-depth comparison.
Direct shallow supervision is a substantive objective constraint;it may help
or interfere with the latent code. The registered fit tests that uncertainty.

These are three fixed checkpoints and a tiny TRAIN sample,not a general
optimization law,quality result or basis for selecting new loss coefficients.
No checkpoint is modified. Disposable qualification weights are not promoted.
