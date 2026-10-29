# Persistent-state cost qualification

The joint-exit probability screen has completed: all three registered gates
fail, and its refined exit does not outperform its supported shallow exit.
No extension or objective-weight sweep follows that result.

The prospective persistent-state model uses six layers, width 512, eight full
attention heads, SwiGLU width 1376, independent 32k input/output tables and
512 real source positions. It has 51,747,328 parameters. The shallower body is
closer to the published small Pythia reference and bounds the activation cost
of an expanded sequence. This is a newly declared shape, not a comparison to
historical ordinary scores. Every future ordinary/pause control must use this
same shape, initialization, input scale, source order and optimizer allowance.

Each source token has one latent slot. Two Jacobi refinements approximate its
causal history; they are numerical solution rounds, **not two thought tokens
or evidence of test-time reasoning scaling**. Fixed two-round full-prefix
readout remains the admitted execution rule until sequential agreement is
measured at learned weights. Only original next-token labels receive loss.

At this shape, dispatched training matrix work is approximately 4.94 times
ordinary training. The trained constant-pause condition is cheaper and its
actual work must be counted. No equal-training-compute win is assumed from
published equal-inference results.

Qualify each execution mode separately on disposable natural TRAIN windows:
two local windows per rank, 64 ranks, FP32, three initialization updates and
100 warm updates. Measure compilation, warm update time, fallback operations,
memory, checkpoint cost and separate-process update 104. Audit the complete
128-window gradient against CPU SDPA/CE. Require the existing 2% family and
0.003 loss limits and exact model/optimizer/RNG replay. No qualification weights
become scientific initialization, and no held-out panel is scored.

Each mode has a separate immutable plan, a 90-minute total cap and a four-GiB
artifact allowance beyond the protected disk reserve. The first admission is
the generated-latent mode. No scientific fit is authorized by this document;
measured cost, appropriate controls, complete cooldowns and a new finite
protocol are required first. Existing failed scientific results stay unchanged.
