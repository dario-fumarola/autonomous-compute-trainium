# Does the fresh-data gain need an additional context read?

The registered source-volume experiment finds a development improvement for
one/two-step branches on fresh views,while repeated views hurt. Two steps do
not beat one. This follow-up must not relabel that as an iteration result.
It asks whether the one-step gain can be explained by a conventional nonlinear
correction at the current position rather than an additional attention read.

Train exactly one new local nonlinear control from the original seed7285,
zero residual output and fresh AdamW. Use the same first8192 TRAIN views,
2048 updates,batch4,32 uniformly sampled natural positions,RNG7286,LR3e-4,
weight decay0.1,clip1.0,and frozen C0/head/features. Do not reuse the trained
attention branch as its initialization. No search,endpoint selection or
extension. All initial512 updates are new work for this different architecture.

The control uses the same2048-to128 input projection and128-to2048 output
projection. Its residual cell is a local128-to512-to128 SiLU feedforward
network with pre-normalization. This has131072 cell parameters,exactly the
same as the attention+FFN cell in the one/two-step models. Total655360 active
branch parameters match. The current-position frozen state already contains
ordinary-model causal context;the control specifically removes the NEW
cross-position read. It is not a context-free language model.

Zero output preserves the exact original predictor at initialization. Check
that changing any other feature position leaves a query's output unchanged,
that every trained parameter receives gradients after the zero-output first
step,and that complete state restoration replays the next gradient/update.
The final full-size branch gradients must pass the same2% FP64 reference gate.
All weights,optimizer moments,counter and target RNG are saved and audited.

Evaluate once on the identical448 reused TUNE views with the unchanged
60/20/20 full32 NLL and source-group interval. Primary descriptive contrast is
local-control NLL minus the saved fresh one-step NLL. Also report frozen C0,
fresh linear and fresh two-step values. Do not infer equivalence from a
non-significant difference or promote a mechanism by choosing a new threshold.
Original two-step0.01/0.005 gates remain failed. This is an attribution test
at one reused seed/panel,not confirmation or a training-efficiency comparison.

Reuse all frozen features without new Native work. Hash-bind them and the
completed comparator scores before launch. Bound CPU training/evaluation to
20minutes and full registration to30minutes including a10-minute mature
audit;measure16 actual updates before retaining the remainder. Use16 CPU
threads,retain512GiB host memory,35GiB disk and10000 inode floors. No unrelated
accelerator job is invented to fill the CPU interval. Count original feature
and comparator work explicitly;do not charge cached representations as free.
No extra context intervention,full-model run,router or confirmation is admitted.
