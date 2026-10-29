# Sequential readout of the selected fixed-round candidate

Register after the complete nine-fit learning-rate screen, before computing
these scores. The best latent learning rate is0.001 under that screen's frozen
K2 selection rule. Keep that checkpoint; do not choose a learning rate or
numerical depth using this follow-up. Its equal-data development advantage is
0.095803nats, but the six-TRAIN-window consistency check fails. This readout
asks how much of the predictive benefit survives the actual sequential history.

Score exactly all384 existing broad DEV windows,512 authentic targets each,
with one prefix-generated continuous latent per source position. Use cached
exact sequential CPU execution in FP64. Every source and its latent share the
source rotary index and occupy separate causal cache entries. No answer-aware
selection, no weight update, no new source selection, no confirmation reserve.
This is the intended sequential history, but differs from the finite-K2
training approximation; that mismatch remains a limitation.

Before full scoring, require the cached and independently fully recomputed
sequential FP64 readouts to agree within1e-8nats per target on the first32
positions of the first two frozen windows in each domain. The previously
qualified CPU reference already passes small-array output/state/gradient and
future-causality tests. Verify the mature audited checkpoint and source hashes.

Report fixed60/20/20 web/math/code NLL and paired source-group comparisons with
the already saved ordinary-lr1, pause-lr1 and latent-lr1-K2 predictions. Those
Native scores passed independent FP64 checks to maximum3.94e-5nats per target.
Bootstrap2000 draws with seed10003. All intervals condition on this selected
single-seed checkpoint and reused development sources: they do not account
for learning-rate selection, adaptive research or training-seed uncertainty.
No efficiency, reasoning, inference-speed or automatic promotion claim.

Four CPU workers, four threads each, batches of two windows, no accelerator
allocation. Bound the entire readout to60minutes from registration and each
worker to8GiB RSS. Preserve512GiB host MemAvailable and35GiB disk on every
output filesystem. Stop and retain partial receipts on any error. Record wall
time and worker memory. Only aggregate results and hashes may be public;
source metadata and per-view scores stay private. This CPU job can overlap
the separately qualified all-chip mixed-round investigation.
