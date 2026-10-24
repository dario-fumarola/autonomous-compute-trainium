# Parcae static-stage global-gradient qualification

Identity: `parcae-native-global-v1`. Numerical qualification, not scientific
training or a reproduction result. No fresh held-out access or terminal alpha.

Prepare independent CPU FP32 objectives while depth-connection-v1 runs; native
qualification must wait until that campaign is terminal and both chip halves
have no managed owner. Do not interrupt or change an existing fit. Bound CPU
preparation to45 minutes and the later paired native captures to30 minutes.

Use the checked Parcae architecture, width768, MHA6 heads, FFN3072,2/2/2 physical
blocks, context1024 and the existing pinned32,000-entry tokenizer/source stream.
The vocabulary is an explicit adaptation from the release's32,768 entries.
Ordinary and recurrent inventories are140,781,888 and141,963,840 parameters;
this is numerical coverage, not a matched-parameter scientific comparison.
The eventual mechanism study still needs a same-parameter depth-one-family
control, a strong ordinary reference and applicable inexpensive connections.

Use first256 existing packed TRAIN windows from the frozen6.5B source, data
seed23 and the unchanged routing271828. No wrapping or generated text. Reference
weights come from the pinned upstream architecture's initialization plus a
declared0.01-scale numerical perturbation, ensuring all maps participate in
backward. These states are qualification-only and cannot initialize a scientific
fit. Seed743, FP32 parameters and primitive/math SDPA throughout. Ordinary uses
one pass; recurrent uses four forward passes and a two-pass backward tail.

Generate and save recurrent initial states privately on CPU, one state per
source window, with pinned hashes. Native capture must read these exact states,
so CPU/backend RNG differences cannot affect the comparison. They are hidden
state fixtures, never source or target data. Publish metadata/hashes only.

Check the port against the independent upstream model on one complete1024-token
training window before accumulating the full CPU objective. Require logits and
full-gradient relativeL2 differences≤2e-5. Save local and complete256-window
CPU FP32 gradients. The latter sum real-token loss divided by the exact global
real-target count, independent of artificial packing targets.

Native capture uses32 ranks per eight-chip half, batch1/accumulation8, DDP
no_sync around forward/backward and the same global real-target normalization.
Compile fixed-shape encode/refine/decode stages, retaining full gradient flow
through the final two recurrent applications. Host loop counts are explicit;
this stage does not qualify sampled-depth training or content-dependent routing.
Keep FP32 masters and disable NKI SDPA before torch import. No optimizer updates.

Gates: full global gradient relativeL2≤0.02 and NLL discrepancy≤0.003 versus
independent CPU FP32; finite gradients; exact target/source/state hashes; every
replica's gradients and unchanged weights agree; no unexpected fallback. Use
separately owned containers,160GiB each /320GiB aggregate, with existing driver,
SDK and XLA preserved. Maintain512GiB MemAvailable and35GiB disk reserves.

Every failure, CPU preparation, graph compilation and capture gets its own
receipt. A pass qualifies this gradient path only. Native optimizer updates,
complete-state recovery, mixed precision, sampled-depth DDP and100 warm updates
remain separate requirements before any scientific run or budget claim.
