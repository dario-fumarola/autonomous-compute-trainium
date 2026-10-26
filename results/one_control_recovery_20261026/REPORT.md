# Original One controls: both original continuations running

Both original seed-17293 One endpoints retain their model, precision, objective,
source/depth order, rate and independent cooldowns: 3,840 equal-data updates and
4,625 work-matched updates. Only their untouched stable update-1,920 checkpoint
is reused. The canceled partial attempts remain separately charged.

A new scientific plan and worker identities preserve the old stopped execution.
Only the checkpoint signature's plan hash changes. Independent disk readback
verifies every other state field, model/optimizer tensor, per-rank RNG and counter
exactly; the original checkpoint remains unchanged. No weights are adopted from
a numerical probe. Future trajectory identity after restart remains unestablished.

The separate precision diagnosis completed and all native owners closed before
admission. The prefix full512-window independent gradient gate passes: body relative-L2
error is0.078928%. Both original continuations now run on all16 chips, after
exact full-state/RNG readback. A fresh
full-512-window gradient capture and independent FP32 gate must pass at the
stable prefix before either fit. Both endpoints then require final gradient gates,
complete-state replica/recovery checks and whole-DEV FP32 scores. The new cap
is at most 9,000 seconds, with the original 09:09:41 UTC absolute cutoff retained.
The complete conservative measured estimate is below that cap. No cheap worker,
new rate/seed, lean qualification, TEST, fresh reserve or terminal alpha follows.

300 CPU tests pass. Both original scientific continuation segments have launched. Finishing these controls
would clarify development replication; it would neither repair the failed cheap
control nor independently confirm the research objective.

[Protocol](../../docs/ONE_CONTROL_RECOVERY_20261026.md) ·
[Migration check](migration.json) · [Plan](plan.json) · [CPU tests](cpu-tests.json).
