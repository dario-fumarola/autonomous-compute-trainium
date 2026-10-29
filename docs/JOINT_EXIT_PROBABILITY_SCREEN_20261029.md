# One joint-exit learning comparison

This registration is executable only after both Native qualification stages
pass, including measured actual-shape cost and mature full-global gradients.

One new79,184,384parameter model starts from the exact untied screen seed9511
initialization. It sees the same200,190,896 natural targets in1530 updates,
with LR.001,32-update warmup,full stable schedule and independent final10%
cooldown to.1peak. The sole intervention is uniform real-token supervision at
the first and final predictions,each weighted.5. The probability proposal
remains fully differentiable and prefix-only. No new parameters,source content,
selected-position emphasis,teacher,router or extra depth is introduced.

The1530-step final-only workspace and both1530/3413-step ordinary controls
already passed their mature gradient,state and readout audits. Reuse their
byte-pinned scores and checkpoint digests with their original costs;this is
one additional optimization arm,not another independent seed. Executed matrix
work is identical to the final-only workspace;ordinary3413 matches within
.024%. Elementwise loss work,compilation,communication and measured chip-hours
are separately reported. This is not a hardware-time match.

Evaluate both supported exits0/1 and one fixed vocabulary-row permutation of
the generated value embedding. Report all broad domains and every existing
source-authored dependency cohort,including weak/negative math and code results.
The384-group broad panel is repeatedly used DEV;the320 protected groups remain
sealed. No new subset or endpoint selection is permitted.

Positive differences favor joint/deeper prediction. Register supported shallow
minus deep,ordinary-data minus deep,ordinary-work minus deep,final-only minus
deep,ordinary-data/work minus shallow,and permuted minus intact. Report fixed
60/20/20 broad NLL,each domain,full authored answer NLL and first-target NLL.
Use the existing2000 paired source-group bootstrap draws(seed9514). Each broad
screening gate requires mean gain>=.01 and conditional95% lower bound>0.
These are development gates,not familywise confirmation or training-seed CIs.
A supported-exit gain cannot substitute for absolute ordinary-model gains;
permutation damage alone cannot demonstrate reasoning. Two-step extrapolation
is not evaluated. No automatic scaling or confirmation follows any outcome.

Retain a recovery-only state at768 and the independently cooled1530 endpoint.
Audit the complete joint training objective over the same256 reserved TRAIN
windows with an independent CPU SDPA/CE implementation;all family thresholds
remain2%,loss.003. The zero-extra exit's gradients were separately qualified
at actual shape;both mature inference exits get independent FP64 checks before
scoring. All64 model/optimizer/RNG replicas must agree,with exact CPU state
restoration. Bound the entire controller to90minutes,each Native phase40minutes
and CPU audit30minutes. Allow4GiB artifacts above the protected35GiB floor;
keep192GiB worker and512GiB host reserves. Failed work remains in the ledger.

This tests whether direct intermediate prediction supervision helps early
optimization. A positive would still require a conventional intermediate-loss
control,independent tuning,fresh source groups and seeds before recurrence-
specific efficiency or thinking claims. A negative closes this objective at
this early exposure;it does not establish a universal latent-computation limit.
