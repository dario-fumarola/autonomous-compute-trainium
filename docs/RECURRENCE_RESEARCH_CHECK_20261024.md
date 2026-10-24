# Literature check while the frozen3B comparison runs

This is interpretation and follow-up triage, not a new experiment registration.
No active architecture, optimizer, source order or endpoint changes.

**Sharing has a measured cost on natural text.**
[How Much Is One Recurrence Worth?](https://arxiv.org/html/2604.21106v1)
reports116 FineWeb-Edu pretraining runs at matched effective depth, a32k
tokenizer and a fitted recurrence-equivalence exponent0.46. Its compute-optimal
looped frontiers lose to ordinary training. The authors separately count
single-use and recurrent parameters and include injection/head work.
This is relevant evidence against expecting a free efficiency gain. It is not
a fitted prediction for our tied-head AdamW models: their optimization,
initialization and iso-depth comparison differ. Their injection ablation also
shows that injection choice can materially affect whether training works.

**Teacher-free natural-source recurrence is already studied.**
[What Makes Recurrence Effective in Looped Language Models?](https://arxiv.org/html/2609.36636v1)
constructs Llama/Qwen-style models from scratch and pretrains on FineWeb-Edu;
the Qwen setup processes about12B tokens. It studies core placement, history
conditioning and timestep conditioning, with outcomes depending on the
inference budget and task. Lightweight conditioning is a plausible future
mechanism, not evidence that our fixed two-pass recipe is efficient. Their
reasoning/knowledge task readouts are not our average natural-text prediction
endpoint. Our claim should concern a controlled benefit, not first discovery
of teacher-free recurrence.

**Attention-only results need their data context.**
[A Controlled Study of Attention-Only Transformers](https://arxiv.org/html/2607.18363v1)
separates parameter, depth and FLOP matching. Its main corpus is explicitly
synthetic and therefore outside this project's current data charter. Its
natural FineWeb-Edu control favors the FFN model by roughly0.04 nats. This
does not justify deleting FFNs or claiming repeated attention will improve our
natural-source compute frontier.

## Consequence for this project

Finish the current paired learning curve and independent audits. Do not call
an equal-data gain a compute win, extrapolate a crossover from stable and
annealed points, or treat a published exponent as our own scaling fit.

Before interpreting a future positive as the value of the extra core execution,
the missing **trained inexpensive reinjection control** remains useful:
one core application followed by the same boundary addition, trained with its
own bounded LR selection. The existing same-weight bypass is an untrained
intervention and cannot substitute for this control. If the cheap trained
path recovers the benefit, the case for paying for the second core weakens.

Any smaller core or conditioning follow-up should be one explicit comparison,
with fresh development choices and the original ordinary/compute controls.
No such new fit is launched by this note. Fresh confirmatory source groups,
new training seeds and the existing repeated-testing ledger remain required.
