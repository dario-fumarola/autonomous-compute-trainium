# Pass-boundary compilation

The isolated attention probe passes all 64 numerical comparisons. Explicit
1024-position attention takes about 1.15 ms forward and 1.70 ms backward;
tiling does not improve it. Rotary precomputation and slot unbinding also fail
to remove the whole-model forward penalty. None is adopted as a speedup.

Test the original model with compilation at three existing boundaries: initial
source representation, shared latent refinement, and final readout/loss. Call
the refinement graph twice. Autograd remains connected across every call,
including all shared-weight and source-embedding paths. There is no detached
state, truncated gradient, extra pass, changed dtype or altered objective.
Ordinary and pause retain their existing single compiled objectives.

CPU eager and AOT-compiled segments must agree with the original full graph in
loss, all parameter gradients and three AdamW updates. Then qualify the actual
51.75M shape on all16 chips: cold and mature independent global gradients,
100 warm updates, all-replica equality and exact separate-process replay.
The early execution screen remains 0.8 seconds over updates4–12. Every weight
is disposable; no heldout panel or scientific fit is admitted. Nominal matrix
work remains identical. Report any runtime improvement only after the complete
qualification, preserving the slow and failed implementation histories.
