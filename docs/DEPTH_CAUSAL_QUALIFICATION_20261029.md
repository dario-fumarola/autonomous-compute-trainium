# Disposable actual-shape depth-causal qualification

This plan follows complete software checks for the reference described in
DEPTH_CAUSAL_WORKSPACE_REFERENCE_20261029.md. The preceding supported-budget
experiment must complete and retain every result before admission. This is a
new graph-feasibility test, with no scientific training or automatic promotion.

Use all16chips/64ranks, local batch2,512source positions, FP32, the existing
six-block51,747,328-parameter model and pinned disposable TRAIN views. Require
512GiB host available,192GiB Native worker limit,35GiB free on each output
filesystem plus4GiB artifact allowance,10000inodes plus both snapshots.

Cold gradients and scalar loss at budgets0/1/2 must agree with independent CPU
full-mask recomputation under the unchanged2%family and0.003nat gates. Then
run104disposable updates cycling budgets0/1/2, with budget2 at update104.
Retain100warm updates4through103; require every budget mean <=0.8seconds and
the early nine-update mean <=1second. No solver or hidden detached cache.

Save every replica's complete model/optimizer/RNG state. Restore update103 in
a fresh Native process and reproduce update104 exactly. Audit its full global
128-window gradient independently on CPU. Include all initialization, compile,
checkpoint, replay, failure and CPU costs; no qualified weights may initialize
scientific training. The clock is45minutes from registration; each Native
phase is capped at20minutes and the CPU audit at15minutes. One attempt only;
no automatic retries or fitted-model work.

Bind source, protocol, data and prior evidence hashes before any update.
Implementation uses the already qualified execution supervisor and warm/replay
engine via depth_causal_entry.py. Numerical fixtures and repeated TRAIN views
are software qualification, never a predictive quality result. Both fresh
reserves remain unscored. Prior failed scientific gates are unchanged.
