# Timing and inexpensive controls: prospective questions

This primary-source check changes no model, schedule or endpoint in the active
three-seed mature-exposure study. No additional arm is registered or launched.

[Elastic Latent Recurrence, sections6.5–6.6 and appendixF](https://arxiv.org/html/2610.12582v1)
reports0.5B dense ablations trained on300B tokens. Delaying mixed depths to
the final quarter reduces estimated total training cost from1.79 to1.20
times its baseline, retaining84% of the five-loop loss improvement. Its
later conditioned variant adds6.34% parameters. One-loop gating itself also
improves loss: the gated one-loop-only model scores1.660 versus1.685 for the
ungated baseline. Thus loop benefits and a cheaper boundary/operator benefit
need separate controls. The cited ablation data is a subset of Nemotron
pretraining data; we have not verified it against our natural-source-only
policy. These comparisons do not establish an ordinary-more-data matched-
training-compute win under our constraint. Their gate, depth schedule,
architecture, exposure and objectives differ from ours.

[Closing the Loop, section3.4 and appendicesA/F](https://arxiv.org/html/2610.00673)
uses a100B response-only mid-training mixture containing explicitly synthetic
sources after210B pretraining. Its conversion study pairs60B continued
training with parameter/inference-size references on instruction-heavy data.
Those results are not a direct test of our natural-text NLL objective or
teacher-free corpus constraint. The normalization appendix also reports
post-norm/sandwich plateaus, whereas pre-norm learns. This is relevant context
for our archived collapse, not a diagnosis of its precise cause.

[From Growing to Looping, section5.3](https://arxiv.org/html/2602.16490)
studies retrofitting recurrence during a changed, higher-quality cooldown and
selects middle blocks. Its reasoning findings motivate timing questions;
changing the data mixture and selecting a block would answer a different
question from our fixed natural stream.

Our inference: paying for recurrence only after ordinary features are learned
could reduce the early training cost. That is a testable alternative to the
original expectation that recurrence must start at initialization, and could
contradict it. Early versus late also changes exposure to recurrence. Timing,
resource efficiency and matched-pause/loop exposure would need separate claims.
This is established prior art, not a new generic late-loop invention.

There is also an inexpensive causal control worth retaining. Train a
parameter-matched one-core model with the same boundary reinjection before
the coda. Our existing same-weight reinjection-only ablation was not trained
that way and cannot replace this control. If such a model improves, it would
identify an architecture benefit without establishing a benefit from another
sixteen-layer core application. A future efficiency confirmation should use a
strong ordinary reference that includes applicable inexpensive changes.

Neither idea authorizes modifying an active fit. Finish the present matched-
endpoint comparison, preserve its outcome, and then choose one bounded next
intervention with a new protocol and measured budget. Do not combine a timing
switch, learned gates, per-pass capacity, depth randomization and new data in
one comparison.
