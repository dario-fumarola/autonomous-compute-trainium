# Three stronger baseline numerical profiles pass

Lean One,corrected lean mixing and conventional ordinary8 all pass the original
full-gradient,actual first optimizer/moment and named replay criteria. Exact
loaded model/optimizer state and rank RNG,replica equality and fallback checks
pass. No qualification-updated weights are adopted for scientific training.

| Profile |100 warm updates|Independent global gradient relative error|
| --- | ---: | ---: |
|Lean One|119.2443s|0.089119%|
|Corrected lean mixing|116.1646s|0.069695%|
|Ordinary8|94.0158s|0.026671%|

The existing242 valid warm-process updates are reused;the recovery makes124
new qualification updates,total366. First-stage failures and the interval before
inspection remain separately accounted. The successful replay criteria do not
establish identical future trajectories after restart.

Predictive comparisons remain outstanding: ordinary8 needs its own bounded
learning-rate tuning and corrected mixing needs new training from original
random initialization,with proper work matching,cooldown and final audits.
No4B run,scientific fit,TEST,fresh reserve or terminal alpha follows automatically.

[Complete result](report.json) · [Costs](execution-costs.json).

The original registration below records the recovery's scope and budget.

# Recover only missing baseline numerical work

The first stage remains stopped. Its current filesystem reproduces an inode
reserve failure; the initiating exception was lost behind paired cleanup.
Regenerable closed-snapshot bytecode is reclaimed, with source/state integrity
verified. The new plan requires24,624 free inodes for four remaining phases,
including the protected10,000 reserve; current headroom exceeds that projection.

Reuse the two completed121-update warm states. Independent full512-window FP32
checks pass for lean One and corrected mixing, with0.089119%/0.069695% overall
gradient errors. Their actual first clipping/optimizer/moment checks also pass.
No qualification-trained weights enter a scientific initializer.

Only new One/mixing replay identities and ordinary8 warm/replay remain:124 new
qualification updates plus242 inherited successful updates. The complete
conservative remaining estimate is2,935.73 seconds under a3,600-second cap.
The numerical thresholds, model operators, precision, source order and
qualification schedule stay unchanged. Source/guard exceptions are retained
before paired cleanup. The old partial identities are not reused.

316 CPU tests pass. This registration does not admit scientific fitting, a
rate sweep, fresh source scoring, TEST or terminal alpha. Ordinary8 still needs
independent tuning and corrected mixing needs fresh training before predictive
comparisons can be made.

[Plan](plan.json) · [Protocol](../../docs/BASELINE_NUMERICAL_RECOVERY_20261026.md) ·
[CPU tests](cpu-tests.json).
