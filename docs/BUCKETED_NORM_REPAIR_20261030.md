# Preserve norm decomposition; repeat the same mature checks

The first mature CPU attempt fails the original1e-5 relative-error bound on
clipped ordinary gradients. Large concatenated FP32 norm reduction changes the
coefficient enough to alter gradients and moments. Its failure and four
disposable steps remain recorded at mature-cpu-v1, source commit106f58d.

Narrow the helper to batching communication, division and final coefficient
application. Retain the existing per-parameter L2 norms followed by their joint
norm, in the original order. Do not relax any error threshold. Add a large-vector
FP32 regression fixture; small arrays alone did not expose the discrepancy.

One new bounded CPU attempt repeats every original mature checkpoint, gradient,
multiplier, optimizer state and comparison, including the failed clipped case.
Bind the original failure receipt. Ten minutes, eight threads and8 GiB RSS;
the same resource floors and complete1e-5 gradient/update/moment gates apply.
All updates are disposable CPU qualification. No scientific checkpoint is
changed, no Native work is admitted, and the active scientific source manifest
must remain unchanged. A pass would still require separate accelerator
collective, state, fallback and whole-update throughput qualification.
