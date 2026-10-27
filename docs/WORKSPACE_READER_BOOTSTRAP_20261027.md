# One bounded reader-interface bootstrap

The zero-update ten-anchor TRAIN qualification shows an input-dependent reader
but severe framing damage: raw/empty-frame/sampled NLL2.7443/5.7576/6.4248.
Before policy training, run exactly one CPU calibration from the same pinned
stage-one base. This is interface qualification, not a mechanism comparison.

Use30 updates on30 other accepted TRAIN source groups,18web/6math/6code.
Each has128 prefix tokens and four natural continuation labels. Generate four
eight-token workspaces per prefix from the initial model,with prefix-only
sampling. Keep those workspaces fixed during this reader-only bootstrap.
Do not optimize the writer by policy gradient or select workspaces by targets.

Train all actor parameters with0.5 ordinary-source NTP plus0.5 workspace-reader
NTP. Raw NTP predicts the observed prefix suffix and continuation;reader NTP
predicts only the same four natural targets from prefix/workspace/framing.
Use AdamW,peak LR1e-5,betas0.9/0.95,epsilon1e-8,weight decay0.1,gradient clip1.
Three warmup updates,24 stable updates,three cosine cooldown updates to zero.
No LR/frame/mixture search or automatic extension follows. Keep all costs.

Evaluate the original ten groups,excluded from bootstrap training,using their
fixed initial samples and separately fresh prefix-only samples. Report raw,
framed and sampled NLL and reward spread. This is held out only from this tiny
calibration,not a fresh generalization benchmark and not proof the upstream
base never saw the text. Better framing can result from ignoring thoughts;
do not label it learned thought. All comparative controls remain necessary.

Save one complete model/AdamW/RNG checkpoint and independently reload it.
Perturb parameters,moments and RNG before loading,then compare every restored
tensor and nested state exactly. Require finite gradients and losses,512GiB
host reserve,35GiB output reserve,and complete20GiB checkpoint space plus1GiB
margin before admission. Use a30-minute operational cap and16 CPU threads.
No accelerator worker or full scientific fit is admitted.

If a later policy discovery uses the calibrated checkpoint,all conditions
must start there and charge the common bootstrap history. Otherwise rerun the
entire discovery from the original base with separately counted preparation.
There is no automatic learned-router or full from-scratch campaign.
