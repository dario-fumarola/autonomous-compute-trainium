# Scientific fit complete; original Native readout stopped

Training, cooldown, mature gradients and full-state restoration pass. Native
readout stopped on the permuted-value condition at 0.000222091 target-loss
error, above the unchanged 0.000200 limit. Its partial scores are not admitted.

The [registered FP64 recovery](../joint-exit-readout-recovery-v1/REPORT.md)
recomputes all six endpoints without further training and closes the scientific
comparison. All three broad gates fail. Keep this original stopped history and
its execution costs alongside the recovery; it has not been overwritten.
