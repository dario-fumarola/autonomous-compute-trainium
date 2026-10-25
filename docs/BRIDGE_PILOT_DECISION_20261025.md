# Decision after the bounded bridge pilot

The recipe is closed negative. It solves its TRAIN cases almost perfectly but
only2/135 One and4/135 R held-out answers. The same R's extra supported pass
adds no held-out successes. Both bridge conditions harm broad-source prediction
by about0.011 nats relative to NTP-only. No longer training,LR/weight sweep,new
favorable subset or reinterpretation of answer NLL follows from this result.

A scope limitation matters for the next decision:to start the adaptation from
identical weights,the pilot initializes both One and R from the existing
one-pass foundation. Recurrence is introduced only during this short200-update
continuation. This tests acquisition of a repeated path during the curriculum;
it does not test whether the already trained recurrent foundation learns the
curriculum better. The before-training readout and original NTP study contain
that separate recurrent checkpoint,with measured benefit from a supported
second pass on ordinary natural text.

The most direct remaining checkpoint experiment is one bounded initialization-
history comparison:retain that existing full-gradient recurrent1B foundation,
compare with the existing work-matched longer One foundation,and otherwise
keep the bridge task/objective/curriculum fixed. Give both an identical fresh
NTP continuation slice beyond either parent's cursor,identical task ordering,
reset optimizers and the same fixed recipe. Include the corresponding NTP-only
continuations and a fully accounted longer One curriculum control. Charge each
model's complete different pretraining history. This would extract value from
existing checkpoints rather than scale a failed curriculum or invent an
architecture. It requires a new frozen plan and per-parent mixed-gradient gates;
this note does not launch or admit it.

The question would be whether pretraining a useful repeated computation changes
what the source-work curriculum can teach. It must not be advertised as proof
that this current pilot was a bug,or as a promise that generalization improves.
If that bounded check also lacks a useful held-out effect and supported-pass
benefit,close this bridge family at this scale. Any domain positive still needs
fresh-seed/source confirmation,and cannot replace the original broad NLL goal.

Separately,the fixed-two-pass small-core plain-NTP control remains untested in
the earlier full-gradient variable-depth study. It is a possible future direct
broad-prediction comparison,with unchanged architecture and properly tuned,
lean One controls. The verified literature budget-regime note is a hypothesis
source,not permission to run an unbounded longer training campaign. Finish one
registered question before choosing between these follow-ups.
