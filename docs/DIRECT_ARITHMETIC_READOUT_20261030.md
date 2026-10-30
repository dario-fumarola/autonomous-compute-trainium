# Frozen readout of direct authored calculations

Read every case in direct-arithmetic-panel-v1 with the fixed, already audited
latent, strongest tied ordinary and trained-pause models from seeds9901,
730201 and730237. No new fit, LR selection or example removal. Require the
source-preparation coverage gate and complete training-prefix/source-group
checks before admission. Both confirmation reserves stay sealed.

The primary score is the mean of per-case answer-token NLL within each of the
three operation bins, then the equal-weight mean of those bin scores. Every
case has a distinct source group. Report first-answer-token NLL and exact
answer-token-prefix accuracy secondarily; termination is not evaluated.
Teacher forcing does not condition the first answer token on any answer token.
All models receive identical unchanged source prefixes and target strings.

Use exact sequential cached latent execution, with vocabulary projections only
at the scored exits. Ordinary and trained pauses use their supported parallel
causal forward paths. These are the trained single-latent operating points;
Jacobi training rounds are not additional thought budgets. No unsupported
depth, generated rationale, prompt alteration or target-aware selection.

For every checkpoint, verify selected readout against independent full-prefix
recomputation on three short prefixes at1e-8 log-probability error. Require all
27 checks. Verify batching against individual readout at the same tolerance.
Preserve all selected-model, training-plan, source and evaluation hashes.

Report all nine cells and both paired-control contrasts for each seed, all
operation bins, macro averages, and conditional source-group intervals using
2000 stratified bootstrap draws, seed40530. Also report the paired-seed SD and
nominal two-sided95% t interval over the three seed differences. This includes
one originally tuned development seed; it is not terminal confirmation. Source
intervals and training-seed intervals must never be substituted for each other.

A useful task-specific development lead requires at least0.03nats over BOTH
controls in EACH of the three seeds, with positive conditional lower bounds.
Failing this rule is a negative/inconclusive diagnostic at this exposure, not a
universal reasoning limit. Passing it would justify a separately registered
content intervention; none is admitted here. No automatic training or scaling.

The latent model used3.782219times ordinary training matrix work at equal real
data. A task lead therefore cannot establish training efficiency or restore the
failed broad replication gate. The arithmetic verifier only selected existing
source targets; it never supplies thoughts or prediction-time inputs.

Use nine CPU workers, four threads each, batch at most8 equal-length cases,
6GiB RSS per worker and30-minute duration. No Native owner may overlap. CPU
costs are reported, with zero accelerator assignment. Any failed worker or
identity check stops the readout; do not exclude it or retry opportunistically.
