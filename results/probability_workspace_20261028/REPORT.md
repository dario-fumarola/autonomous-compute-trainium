# Probability-embedding replay: qualification, not a quality result

An independently implemented PonderLM-style operation forms a probability-
weighted input embedding from a causal prediction,adds it to the input stream,
and recomputes the complete causal body. Ordinary NTP differentiates through
every pass. This is distinct from the closed sparse raw-hidden-state append and
frozen-feature correction mechanisms. No positive published result is inherited.

Seven focused CPU tests pass:exact zero-step ordinary outputs/gradients/AdamW,
independent expected-embedding gradients,causality across replay,independent
SDPA-versus-explicit attention gradients,correct recurrence and work accounting.

The all16-chip/64-rank small-array fixture passes all five disposable updates
and an independent-process replay. Maximum overall gradient discrepancy is
2.911e-6 relative;maximum optimizer-update discrepancy1.054e-4. Full model,
optimizer and RNG replay is exact;replicas agree and no unsupported fallback
occurs. Assigned qualification cost0.610 chip-hours;controller opportunity0.675.
There are zero research examples or scientific updates. The unlaunched original
registration is preserved alongside the preactivation RNG-alignment amendment.

Next is actual62.8M-shape numerical/throughput qualification at512 context,
using only accepted TRAIN data. Its first preparation attempt stopped on the
projected inode allowance before creating a plan or launching hardware;an
inadvertent run invocation also stopped immediately because no plan existed.
Bytecode-only maintenance repairs admission;these stops are not model outcomes.
No scientific fit,quality comparison or full from-scratch campaign is admitted
by the small-array result.

The source preparation retains638886402 authentic targets in a640M-position
stream,60/20/20 web/math/code,with384 tuning source groups. Training/tuning and
the320 protected groups have zero overlap. This is a new bounded source ordering
with the original32k tokenizer,not OLMo-tokenizer continuation or fresh terminal
confirmation. All text and token arrays remain private.
