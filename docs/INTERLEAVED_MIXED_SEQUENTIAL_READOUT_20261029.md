# Exact sequential check of the selected mixed-round endpoint

This is prepared before mixed common DEV readout and admits no fit. After the
complete mixed LR comparison and unchanged selected-checkpoint consistency
gate pass, evaluate its K2-selected checkpoint once on all384 existing broad
DEV windows, under exact cached sequential generation. Selection is fixed by
the original K2 criterion; this readout cannot change the selected LR.

Use the same authentic targets,60/20/20 domain weighting, FP64 cache computation,
and six cached-versus-full-prefix parity checks as the prior sequential receipt.
Compare against the retained selected ordinary and pause controls, and the
same selected mixed weights at supportedK2/K3. Report every contrast and each
domain. Source-group intervals use2000 bootstrap draws, seed10003. They do not
include selection uncertainty, training-seed variation or repeated DEV use.

The TRAIN guard samples six windows and cannot establish full-DEV consistency.
This larger readout directly measures deployed sequential quality instead of
assuming the approximation is accurate everywhere. It does not test multiple
thought-token budgets: there remains one generated latent state per source token.

Bound to60minutes, two CPU workers of eight threads,4GiB RSS each. Zero optimizer
updates and accelerator chips; preserve512GiB host and35GiB output reserves.
Source, checkpoint, original audit, existing control score arrays and panel
hashes are frozen at admission. Both fresh reserves remain closed. No automatic
main, learned router or confirmation follows from this result.
