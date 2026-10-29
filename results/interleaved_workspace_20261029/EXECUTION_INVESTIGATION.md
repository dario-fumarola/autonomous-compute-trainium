# Persistent-workspace execution investigation

No quality result or scientific fit is produced by these measurements. The
original 51.75M-parameter implementation passes cold/mature gradients and exact
full-state replay, but costs 4.4902 seconds per warm update against 0.1459 for
ordinary training. Its nominal matrix-work ratio is 4.937, not the observed
30.78 elapsed-time ratio. The excess motivates execution diagnosis.

| Check | Observation | Decision |
| --- | --- | --- |
| Original full-model stage timing | Forward4.138s, backward0.063s, collectives0.282s | Investigate forward execution |
| Slot-axis unbinding | Cold gradients pass;4.487s per update | Reject speed hypothesis |
| Precomputed rotary positions | Cold gradients pass;4.469s per update | Reject speed hypothesis |
| Compile at pass boundaries | Cold gradients pass;roughly4.44s per update | Reject speed hypothesis |
| Attention operator fixtures | Numerically correct,1024-position forward about1.15ms | Isolated attention is not slow |
| Runtime/device profile | Confirms long forward; instruction capture has no usable events | No operator-level attribution |
| Pass-profile v1 | Qualified-state gradients agree; unexpected fallback stops final report | Retain failed diagnostic and its costs |
| Pass-profile v2 | Reporting recovery within v1 deadline; timings/fallbacks persisted each batch | Diagnostic only; unexpected fallback cannot qualify an implementation |

The slot-layout CPU trace attributes a synchronization wait to a copy. That
does not establish that copying causes the forward penalty. Likewise, device
trace analysis failed to match cached binary UUIDs and must not be cited as
instruction-level evidence.

The segmented diagnostic reports `aten::contiguous[prologue_cpu_roundtrip]`.
That path is not adopted. Its fallback does not retroactively alter the
original whole-graph qualification, which recorded no unexpected fallback.

Numerical fixtures use random arrays solely to test software. Full-model
qualification uses authentic TRAIN windows with disposable weights. No held-out
scores, fresh reserve access, external teacher, synthetic research examples, or
scientific optimizer update occurs. All stopped work remains in the linked
per-attempt execution receipts. Model/scorer arrays and raw runtime traces stay
private.

The full grouped implementation subsequently passes independent cold/mature
checks,100 warm updates, all-replica equality and exact fresh-process full-state
replay. Generated-state warm mean is0.246338s (18.23 times faster than the
original layout), pause0.182445s. Maximum mature gradient-family errors are
1.09e-6 and2.43e-6. This preserves the mathematical computation, not bitwise
optimizer trajectories across differently ordered floating-point reductions.
The original chronological CPU reference remains independent.

[Generated qualification](latent-grouped-v1/report.json) ·
[Pause qualification](pause-grouped-v1/report.json) ·
[Join fixture and compiler-failure recovery](join-probe-v2/report.json) ·
[Original failed join fixture](join-probe-v1/report.json).

All709 tests pass with two skips. A new finite LR-selection campaign is
registered separately. No experiment above establishes a prediction advantage.
