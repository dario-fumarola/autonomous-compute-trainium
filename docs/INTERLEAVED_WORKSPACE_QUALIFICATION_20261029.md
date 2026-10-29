# Persistent latent-token qualification

This is a prospective software and numerical test, not an admitted learning
campaign. It does not replace the pending joint-exit readout or its negatives.

One generated final hidden state is interleaved after each original source
embedding. Original and latent slots share the original rotary position. Two
fixed Jacobi refinements update the latent from the preceding source slot;
the final expanded pass predicts the authentic next token from each latent
slot. All passes receive gradients. Each real target receives one NTP loss.
The input scale is sqrt(2.5 times width), as in the pinned hidden-feedback
reference. Output rows remain independent of input rows. Ordinary and trained
constant-pause paths use identical parameters and source scaling.

Reference: [PonderLM-2](https://arxiv.org/html/2509.23184v4),
[pinned implementation](https://github.com/LUMIA-Group/PonderLM-2/blob/fa784bece621b989fb008c59b0fd8d282fa9c73c/src/llamafactory/model/modeling/modeling_gpt_neox.py).
This is an independent Llama-style implementation, not a reproduction. The
paper samples two or three refinements; released code samples two through four.
This qualification fixes two. The architecture, tokenizer, initialization and
data differ. The published headline comparison matches inference, not training
work. No published quality advantage is inherited by this variant.

The central distinction from probability replay is that generated states occupy
causal history slots which later tokens can read. It is still not evidence of
reasoning. Full-prefix Jacobi evaluation is not assumed equivalent to cached
sequential inference at only two refinements. Short fixture tests prove exact
forward AND backward equivalence at convergence and measure finite-round
stabilization. A learned-state convergence diagnostic would be necessary before
any cheaper sequential-deployment claim.

CPU gates: future invariance, exact ordinary path, gradient flow through earlier
latent states, independent SDPA versus explicit attention, masked source-target
coverage, exact optimizer recovery, and dispatched forward/backward matrix work.

Next qualification only: all 16 chips on small disposable arrays, independent
global CPU gradients and first optimizer moments, 2% family error and 0.003 loss
limits, all-replica equality and separate-process exact next-step replay. No
natural model scores, generated research examples, or scientific fit. Actual
shape cost/memory qualification and a separate finite registration must precede
any learning run. Preserve 512 GiB host, 35 GiB disk and inode reserves.
