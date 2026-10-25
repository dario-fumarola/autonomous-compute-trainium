# Fixed baseline readout before a curriculum fit

The task driver passes all six independent full32-example FP32 gradient/
optimizer checks with maximum relative-L2 error0.08803%. Its qualification is
not a science result. The source-view pool has1,618 deduplicated TRAIN problems;
the new discovery DEV panel has135 original tasks across115 unused groups.
All students and held-out outcomes remain unread at registration.

Before tuning or fitting the curriculum,measure existing plain-NTP one-pass
and recurrent endpoints for seeds5231/7283 on this same new development panel.
Use the completed1.005B checkpoints,remove only their training-only auxiliary
head and leave every prediction parameter unchanged. The one-pass checkpoint
is evaluated at its supported depth1;the recurrent checkpoint at predeclared
supported depths1,2,4. No best-depth selection or terminal claim follows.

Use only the no-work view at inference. Greedily generate at most24 new tokens,
with an additional cap based only on visible prompt length and context128.
Never use original answer length,answer tokens or hidden work to decide inputs,
termination,budget or exit. Stop on EOS. Count a scalar answer correct only if
EOS is reached and its entire decoded scalar matches the original source
answer under the bounded rational verifier. Fixed English task delimiters are
shared. Special symbols/prose/calculations as outputs are invalid,not repaired.

Initial latent noise is deterministic from the visible prompt token IDs only,
with the existing clipped-normal scale;reuse it across inference depths and
all generated tokens. No input-dependent routing is learned. Report whole-
answer accuracy,termination,first-answer-token accuracy,original answer NLL
and actual generated tokens/forward calls. Teacher-forced NLL uses the normal
causal decoder and is scored only after free generation for each example.

Independently audit full-panel answer NLL using pinned upstream FP32 forward
arithmetic. Require maximum aggregate difference0.003 nats. Predictions and
source strings stay private;public aggregates retain immutable hashes. Check
prediction weights/replicas unchanged. Native full-window recomputation is
uncached;these timings do not establish production serving efficiency.

Use all16 chips as two owned eight-chip jobs per wave,two waves for the four
checkpoints. Cap each native phase at600 seconds,CPU audit at600,overall1,800.
Preserve512GiB MemAvailable and35GiB disk. Charge failed attempts and CPU tails.
No fit,TEST or terminal alpha. This makes the explicitly designated DEV panel
consumed for development. Subsequent study choices must disclose that reuse.

This is a before-training capability readout. A curriculum experiment still
needs matched One/R training on the same source-derived task views,ordinary NTP
retention,learning-rate opportunity and a longer One work control. A continuation
pilot cannot establish from-scratch or general-text efficacy;those original
requirements are preserved.
