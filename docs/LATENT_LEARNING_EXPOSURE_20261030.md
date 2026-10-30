# One bounded learning-exposure experiment

## Question and scope

Does the existing dense, generated-state mechanism acquire a useful natural
calculation advantage after substantially more ordinary pretraining? The128M
screen did not establish this: its direct-arithmetic result varies in sign
across seeds, with near-floor accuracy. Its robust pause-control broad gain is
evidence worth investigating, but its ordinary-control replication gate failed.
That failure stays recorded. This is a new exposure question, not confirmation
or automatic promotion of the original result.

The source census finds617 intact verified calculations in the128M prefix and
4,398 in the prospective1B prefix. This supports testing exposure as an
alternative hypothesis; it does not prove that exposure caused the failure.
The complete ordinary matched-work prefix is also audited before training.
Ninety-four historical evaluation expressions occur in that longer prefix.
The fixed primary arithmetic subset excludes them:259cases, from259 source
groups, with87/98/74 cases in the2/3/4–6-operation bins. The original353-case
result remains unchanged. Neither fresh confirmation reserve is opened.

Published methods provide context, not a promised effect: PonderLM-2 includes
410M-model experiments on6B tokens, and adaptive latent-CoT pretrains410M and
1.4B models on26B tokens. Our51M/128M screen is a much earlier learning regime;
these publications do not establish an exposure or size threshold for this
experiment. [PonderLM-2](https://arxiv.org/html/2509.23184v4) ·
[Adaptive latent CoT](https://arxiv.org/html/2602.08220v2).

## Frozen scientific conditions

Use one new initialization seed730271. Every fit starts from random weights;
do not extend an already cooled checkpoint. Keep the qualified architectures,
tokenizer, indexed60/20/20 source order, uniform authentic-target NTP, AdamW,
weight decay0.1,64-update warmup and complete final10% cosine cooldown. Preserve
the independently selected128M learning rates, all0.001, deliberately testing
fixed recipes at a new exposure. They are not independently retuned at1B;
no optimal conventional frontier or final efficiency claim follows.

| Condition | Parameters | Real targets | Updates |
|---|---:|---:|---:|
| Generated persistent state, one latent per token |51,747,328|1,000,060,368|15,287|
| Strongest tested tied ordinary, eleven unique layers |51,179,008|1,000,060,368|15,287|
| Trained constant-pause control |51,747,328|1,000,060,368|15,287|
| Conditional longer ordinary, matched matrix work |51,179,008|3,782,650,274|57,822|

Latent training uses a frozen balanced K2/K3 Jacobi schedule with seed9917.
This changes numerical solver rounds, not the number of thought tokens. All
gradients are retained. Ordinary uses its qualified K0 path; pauses retain
their qualified path. Generated and pause initial weights must match exactly.
Keep an untouched pre-cooldown recovery state plus the final state for each
fit. No intermediate quality score chooses duration, LR or initialization.

All three equal-data fits complete their final independent128-window gradient
audits, full model/optimizer/RNG restoration and64-replica checks before any
quality readout. Numerical-family limits remain2%, with the existing loss
tolerance. No failed arm may be dropped or retried under this plan.

## Readout and finite decisions

Use all384 reused broad DEV windows and all259 source-audited arithmetic cases.
Score latent predictions by exact sequential cached execution in FP64. Retain
the existing independent full-prefix checks. Ordinary and pause predictions
use their supported paths. Arithmetic scoring uses the same checked batched
selected-position implementation as the completed nine-cell diagnostic.

Report the60/20/20 broad NLL and every domain. Arithmetic primary is the mean
case answer-token NLL within each operation bin, then the equal-weight mean of
the bins. Report first-answer NLL and exact answer-prefix accuracy separately.
Use2000 conditional source-bootstrap draws with seed40531. One new seed and
reused development data cannot establish training-seed uncertainty or a final
reasoning claim. No prompt search, unsupported depth or answer-aware selection.

Only if the generated model beats BOTH equal-data controls by at least0.01
broad nats and0.03 arithmetic nats, with positive conditional lower bounds for
both metrics and nonnegative first-answer NLL gains, run the already specified
ordinary matched-work condition. Otherwise close this exposure screen without
launching that longer control. This is a resource-saving development gate,
not an inference that a failed gate proves the mechanism impossible.

If the longer control runs, require the same margins against it. Publish its
result even when it removes the lead. No additional seed, model size, LR or
duration is admitted here. A favorable result justifies a separate content-use
and replication design; it does not demonstrate allocation, useful additional
supported thought steps, causal reasoning or a terminal efficiency result.

## Resources

The computation graph, training worker and optimizer are unchanged from their
completed numerical qualifications. Revalidate their hashes and original
mature receipts before admission; test the new budget/controller/statistics
logic. Use all16chips/64ranks, one exclusive Native owner at a time. Expected
retained fitting time is roughly4hours for the three equal-data runs and a
further5hours only if the longer ordinary gate passes; these are measurements
scaled from prior fits, not new timing results.

Freeze a12-hour queue cap,2-hour Native cap per equal-data fit,6-hour cap for
the conditional ordinary fit,30-minute mature CPU audits, and existing bounded
CPU readouts. Preserve512GiB host,192GiB worker,35GiB output-disk and projected
10,000-inode reserves; require9GiB additional NVMe artifact capacity. Charge
all startup, compilation, failed work, audits, readouts and source preparation.
Stop on an initiating exception and preserve it. No silent restart, deadline
renewal, new spending tier or purchased infrastructure follows.
