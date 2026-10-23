# Prospective read-only diagnostics — 2026-10-23

Declared during the LR screen, before any main seed103/211/307 fit or curve
readout. This annex adds measurements of existing checkpoints, not models,
training, LR choices, primary contrasts, success criteria or TEST cells.
The main protocol and its frozen source hashes remain unchanged.

After the primary release freezes, independently verify the mixture models'
full fresh DEV scores at depths1 and3 for all three1B seeds, using the separate
FP32 CPU implementation. Check every real target and unchanged checkpoint
hashes; require CPU/native weighted-NLL difference <=0.0002. These are numerical
audits on the same DEV data, not new statistical replications. CPU scoring may
overlap the already frozen TEST execution. No TEST targets are used by this
additional audit. Start only if at least50 minutes remain before04:15 UTC.

After the primary campaign closes successfully and releases its accelerators,
score both fixed and mixture main checkpoints at updates384 and1152, depths1
and3, all three seeds. This is24 predeclared DEV cells. Canonical depth2 curves
already belong to the main campaign. All cells retain the same weights,
positions, documents and real targets. Fixed depth1/3 are interventions outside
its training support; mixture depths1/2/3 are within support. These curves ask
when additional passes become useful and must not replace the primary endpoint
or turn an early checkpoint into a selected winner.

Require at least25 minutes for the native diagnostic matrix; otherwise mark it
unexecuted by its time gate. Use the same two eight-chip readout slots and frozen
evaluator. All extra CPU/native work enters the cost ledger. The original hard
deadline04:15 UTC applies. No model update, new architecture, teacher, synthetic
corpus, inference exit selection or additional TEST access is authorized here.
