# Existing authored arithmetic: fixed-checkpoint readout

Before scoring, freeze the mixed-round persistent model, strongest tied ordinary
control and trained pause control from the original seed 9901. Select their
already fixed broad-NLL learning-rate winners. Do not use a new-seed replication
checkpoint or select another model after this diagnostic.

Use all 253 cases in the existing retokenized dependency panel: 134 authored
arithmetic conclusions, 110 exploratory intermediate expressions and nine
code-name targets. Their source groups were excluded from every compared fit.
Keep every original target, prefix and source group. This is reused development
data, not a fresh confirmation or a newly mined reasoning benchmark. The small
code family cannot establish a general code capability.

The primary diagnostic is whole-answer, real-token-weighted NLL on all 134
conclusions. Report control NLL minus latent NLL, so positive favors latent.
A positive development lead requires at least 0.03 nats
and a positive lower conditional 95% source-group interval against BOTH tied
ordinary and trained pauses. Use 2,000 paired source-group draws, seed 40303.
Report failures and uncertainty. No slice replaces a failed broad claim or
changes the inconclusive bracket-type result.

Also report first-target NLL, token accuracy and all-answer-token argmax
accuracy. These teacher-forced metrics do not measure unrestricted problem
solving or termination. Report the other two families separately, without
promoting a favorable exploratory family into the primary.

For each model compare the original full prefix with only the last 32 prefix
tokens, retaining every teacher-forced answer token. Full-context benefit is
not itself a computational mechanism or an advantage over another model.
The supplied original worked steps make this a test of using authentic
arithmetic text, not inventing an unprovided solution. Truncation changes many
features and is not an isolated intervention on arithmetic premises.

Use the exact cached sequential latent path in CPU FP64. The selected-readout
optimization omits unused vocabulary projections only; all source and latent
states remain. Require six actual-checkpoint comparisons with independent
full-prefix recomputation before scoring, at maximum log-probability error
1e-8. Ordinary and pauses use their supported trained paths. Check parameters
unchanged, all files and endpoint hashes, and complete case coverage.

One CPU process/eight threads, four-GiB RSS cap, one-hour immutable deadline;
retain 512-GiB host and 35-GiB filesystem reserves. Zero chips, zero updates,
no fresh reserved group, no terminal alpha and no automatic promotion. This
readout may run while the independently frozen new-seed fits occupy the chips.
