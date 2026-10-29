# Sparse persistent workspace: CPU reference only

The persistent one-state-per-token model has a replicated advantage over
trained constant pauses, but an unreliable useful margin over strongest
ordinary and expensive numerical solver passes. The exact depth-causal
multi-budget alternative removes that cost but fails its quality gates.
Neither result is reinterpreted here. The separately registered fixed-one
comparison remains unchanged.

This reference asks whether sparse feedback can preserve the persistent
state dependency while allowing exact, differentiable cache reuse. Process
a contiguous chunk of authentic tokens. At its boundary, take the last
normalized hidden state as the input to a latent slot and process it through
the shared body. Later authentic tokens can attend to that slot's cached
keys and values. The final latent output predicts the next authentic token
at the boundary; ordinary positions use their ordinary outputs. Every
authentic target appears exactly once in uniform NTP, with no thought labels.

Only full chunks insert slots. An incomplete terminal chunk does not invent
a boundary, which preserves prefix consistency. Source and latent positions
share the boundary's rotary index, while chronological attention order still
prevents a source position from seeing a later latent at that same index.

The reference uses the existing strongest tied-table ordinary family. It
adds no parameters, iteration cues, vocabulary samples, router or solver.
Zero workspace calls the exact existing ordinary path. A constant-input
pause mode is included as a prospective trained computation control. These
modes have software checks, not fitted predictive results.

Two independent implementations agree in logits, latent states and every
parameter gradient: differentiable chronological KV reuse and full-prefix
recomputation. Tests cover FP64/FP32, grouped-query heads, exact disabled-path
AdamW moments, future isolation, partial chunks, cache-gradient faults and
static eager-backend full-graph compilation. Generated states affect later
real-token outputs in numerical fixtures; that checks wiring, not useful
reasoning. Fixtures never enter a research corpus.

## Accounting and the unresolved execution question

At the existing 11-layer, width-512, 51,179,008-parameter shape and 512-source
context, one slot every 128 source tokens stores 516 positions and makes eight
sequential body calls. Its nominal dense matrix work is about 0.58% above
equally chunked ordinary execution, with one vocabulary projection for each
real target. This is an accounting result only.

Chunked ordinary itself avoids some masked attention rectangles present in
the full-sequence dense ledger. Any comparison must report both ordinary
implementations; those saved rectangles are not a workspace benefit. Serial
calls, kernel launch overhead, cache movement, communication and activation
storage can dominate runtime. No hardware advantage follows from the small
matrix-work increment.

This is a deliberate sparse variant of continuous hidden-state feedback,
not a reproduction or a novelty claim. The relevant prior persistent-state
family is [PonderLM-2](https://arxiv.org/html/2509.23184v4).

No scientific fit is admitted. Before considering one, freeze a bounded
actual-shape hardware qualification that compares full ordinary, chunked
ordinary, generated slots and constant slots, verifies complete gradients
and state recovery, and measures whole updates. A positive software check
does not justify a router, reserve access, a larger model or an efficiency
claim. Any learning screen must start with the strong conventional family,
independent tuning and a trained constant-slot control.
