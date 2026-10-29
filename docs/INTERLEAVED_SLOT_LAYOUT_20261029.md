# Equivalent slot-layout screen

The actual-shape graph is correct but slow. A stage-timing diagnostic exactly
replays its qualified update 104, then finds approximately 4.14 seconds forward,
0.063 backward, 0.282 collectives and 0.009 optimizer per update. Synchronization
is explicit; these stage means supplement the original 4.49-second throughput.

One implementation-only hypothesis is that strided source/latent extraction
is expensive in the Native forward graph. Replace expanded[:, ::2] and
expanded[:, 1::2] with a reshape to [batch, source, slot, width] and unbind the
two-element slot axis. There is no change in parameters, latent values,
positions, masks, depth, objective, precision or nominal matrix work. CPU
forward, all gradients and three AdamW updates agree in every execution mode.

Qualify the generated-latent actual shape on all 16 chips. Independent CPU
references retain the original implementation. Require the same cold and mature
global gradients, all-replica agreement and exact separate-process replay.
Measure 100 warm updates; a CPU-only trace on update 2 is outside that window.
Stop at update 12 if the latest nine updates average more than 0.8 seconds.
That is an execution screen, not a loss or scientific stopping condition.
All updates are disposable. No held-out sources or scientific fit are admitted.
The original qualification, its slow timing and all costs remain recorded.
