# Actual-shape Parcae optimizer qualification

Identity `parcae-native-opt-v1`. This is an optimizer numerical experiment,
with prescribed gradients, not pretraining or a scientific gain. No held-out
access, source generation, teacher or terminal alpha.

First prepare independent CPU vectors using the pinned upstream MuonAdamW
algorithm and the complete natural-TRAIN gradient vectors already computed for
the ordinary/recurrent Parcae qualification points. Preserve actual32k-vocabulary
inventories,140,781,888/141,963,840 parameters, FP32 masters/moments and the
released BF16 five-iteration polar transformation. Gate grouping explicitly
follows released code (`release_muon`), including value-embedding gates in Muon;
this does not silently resolve the discrepancy with the paper's grouping table.

Use three numerical optimizer steps with learning rates0.008/0.0056/0.00392,
momenta0.85/0.90/0.95, cautious matrix decays0.20/0.16/0.12 and prescribed
gradient multipliers1/0.75/−0.5. Non-matrix AdamW uses(.8,.95), epsilon1e-10,
zero decay. The signed/multiplied gradients are numerical fixtures that exercise
state and cautious decay; they are not training data or real optimization steps
on the changed model's objective. No clipping is included in this isolated test.

The CPU candidate must agree with the independent released algorithm within
2e-5 relative to the **update size**, with complete named moments within2e-6.
Save every expected weight/moment vector privately and publish only hashes and
aggregate errors. Bound preparation to20 minutes and keep every failed attempt.

Only after the queued native full-gradient gate passes and all managed native
workers release their chips, launch ordinary and recurrent optimizer checks
on separate eight-chip halves,32replicas each. Use fused unscaled AdamW and
explicit compiled Muon tensor updates; no optional AMP metadata. Each does
three prescribed updates, serializes complete model/optimizer state after step
two, and records an uninterrupted third step. A fresh process/container then
loads the saved state and repeats step three. Require exact complete state on
load and identical complete states across replicas. No promise of a bit-identical
future trajectory: report the resumed-versus-uninterrupted update difference.

Native update error, normalized by the independent CPU update, must be≤2% for
each step and≤2% for its complete named moments, both overall and separately for
AdamW and Muon so the embedding update cannot hide matrix-update error. Check
the primary arithmetic before spending work on replay. Apply the same gates to the
fresh-process third step, and require its difference from uninterrupted native
execution≤2e-4 relative to that update. Exact target/source-gradient hashes,
finite vectors and no unexpected fallback beyond `aten::_to_copy` are mandatory.
Source files and reference artifacts freeze before launch.

Latest phase launch02:00UTC October25; all captures, replay and comparisons
must finish within30 minutes of the first launch. A failure cancels this paired
qualification's workers and leaves unrelated owners alone. Each container has160GiB host cap,320GiB aggregate;
retain512GiB MemAvailable and35GiB free on every output filesystem. Preserve
the SDK, installed XLA and drivers. Waiting reserves zero chips.

Count three initial plus one replayed optimizer calls per replica, per arm,
separately from zero research updates. Record CPU preparation, cold optimizer
compilation, prescribed step times, checkpoint/restart and failed attempts.
These timings cannot establish full-training throughput. After a pass, actual
sampled-depth forward/backward training, clipping, complete trainer recovery
and100 warm updates remain necessary before choosing a research budget.
