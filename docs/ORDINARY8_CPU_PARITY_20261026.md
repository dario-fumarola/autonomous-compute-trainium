# CPU check of the conventional eight-layer control

The first preparation is preserved as a failure before any forward or saved
weights: the convenience upstream helper retained its six-layer release
default. Version two explicitly replaces `n_layer` before construction,
including rebuilding the release's depth-dependent initializer. The intended
eight-layer geometry, source windows and numerical thresholds are unchanged.

The metadata inventory identified one conventional control from the parameter
constraint, before observing its quality: eight unique layers, width 640,
five 128-dimensional heads, 4x MLP and the existing alternating value tables.
It has 141,733,120 parameters, 0.1625% below the recurrent candidate's cap.
This changes the width/depth/lookup allocation; it is not a parameter-identical
paired model or evidence of an improvement.

Check this one geometry against the clean pinned upstream implementation on
CPU FP32. Use only the first four authentic TRAIN windows from the existing
packing, fixed before scoring. No DEV/TEST, synthetic corpus, teacher, generated
solutions, architecture sweep or learning-rate selection. The original main
comparison and conditional lean queue remain unchanged.

Initialize the upstream ordinary model from random qualification seed 1009.
Save its unmodified state privately with `qualification_only=True`. Check exact
parameter count, strict tensor-name loading and serialized state readback. The
local candidate uses the existing explicit FP32 attention primitives, with
autocast disabled for this CPU comparison.

Compare ordered real-target NLL, logits and complete named gradients between
upstream and the local implementation on the same four windows. Use a
0.00002-nat loss, 0.0002 maximum logit and 0.2% complete/family gradient gate.
Then apply the same fixed 0.01-times observed natural-reference gradient to
both models and repeat the checks. This numerical state intervention exercises
paths that start behind zero output projections. It is not optimizer or
learning qualification. Count two manual qualification-only model updates;
save neither updated model and never adopt it for science. Verify the saved
original initializer's checksum again at the end.

Use zero accelerator chips, a 900-second CPU cap, at least 544 GiB available
memory and the normal 35 GiB filesystem / 10,000-inode reserves. Publish only
source, counts, hashes and small aggregate errors; retain weights privately.
Native FP16 gradients, actual optimizer updates, full 512-window normalization,
state recovery, throughput and scientific quality remain unqualified.

A later conventional-model pilot needs those numerical gates, independently
selected learning rates and its own measured frozen budget. Preparation does
not admit a native job or a scientific fit. It spends no terminal alpha and
does not access confirmation reserves. The current research goal is unachieved.
