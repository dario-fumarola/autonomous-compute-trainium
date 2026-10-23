# Stopped sandwich-normalization screen

All six 100M-target tuning trials finish at 7.335–7.337 weighted NLL. Both
single-pass and two-pass sandwich paths fail to learn useful context-dependent
prediction under this initialization/optimization recipe. A natural-prefix
readout over 1,536 positions finds KL to the average output distribution near
4–6e-6, versus 3.07 for the healthy ordinary pre-norm reference.

Native CPU/gradient, replica-equality and optimizer-recovery checks passed;
those checks alone did not establish that the architecture was learning. The
main three-seed 1B fits were stopped before any update. All six tuning states
and failed/stopped receipts are retained. This is a failed recipe screen,
not evidence against every sandwich or recurrent architecture. The suspected
normalization/initialization interaction has not been isolated causally.

The subsequent pre-norm study has a separate identity and frozen protocol.
Its single-pass path is mathematically and numerically equivalent to the
preserved healthy ordinary control. No failed tuning checkpoint initializes it.

[Tuning results](tuning.json) · [Natural-prefix diagnostic](context-diagnostic.json) ·
[Stop receipt](completion.json) · [Original protocol archive](../../docs/RECURRENCE_SANDWICH_SCREEN_20261023.md).
