# Credit through the parallel state solver: no-update diagnostic

The full-gradient billion-target model remains unchanged. Its parallel state
solver consumes most of the nominal training work, while removing forward
refinements visibly changes predictions. A different question is whether the
early refinements' gradient contributions are large or directionally important.
This readout diagnoses that question; it does not admit truncated training.

Use only the audited seed-730271, update-15,287 latent model, both trained solver
counts K2/K3, and the original six frozen TRAIN consistency windows. The windows
remain outside the full prospective 3.783B-target fitting prefix. Compare:

1. All original gradients, including the initial source-body estimate.
2. Every original forward, with only the last refinement and final readout
   differentiated. The earlier state estimates are constants for this update.
3. Every original forward, with only the final readout differentiated. This
   removes direct credit to the generated input; shared body parameters still
   receive readout gradients. It is not equivalent to a learned writer update.

Keep natural-target loss uniform within each window. Aggregate the six gradients
with fixed 60/20/20 domain weights. Report cosine, norm ratio and relative L2
difference against full gradients, for all parameters and embedding/body/head
families separately. Keep the full forward logits identical within 0.0002 and
verify complete finite gradients. Software fixtures compare the all-gradient
path to the original, suffix gradients to an independently frozen-prefix graph,
and finite differences with that prefix fixed. Do not claim a shortened gradient
is the derivative of the unchanged full unrolled function.

Report analytic full-forward plus retained-backward matrix counts. These omit
elementwise/optimizer costs, do not measure hardware execution, and do not show
quality after training. Gradient alignment on six windows is neither an optimizer
trajectory guarantee nor a substitute for learning. There is no score-selected
threshold, optimizer update, fresh-reserve read, new fit, or automatic promotion.

One CPU attempt, 20 minutes, 16 threads, 16 GiB RSS, existing 512 GiB host and
35 GiB disk floors. Retain errors and costs. The main ordinary comparison stays
on all 16 chips and retains its original scientific recipe and deadline.
