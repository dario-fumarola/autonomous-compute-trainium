# Fixed-recipe replication: useful ordinary-control margin does not repeat reliably

Both new initialization seeds finish their frozen schedules, full cooldowns,
mature gradient/state audits and exact sequential readouts. No LR was retuned.
Both fresh evaluation reserves remain unscored.

| Seed | Generated latent NLL | Strongest ordinary NLL | Trained pause NLL | Gain over ordinary | Gain over pauses |
|---|---:|---:|---:|---:|---:|
| 730201 | 3.280123 | 3.282390 | 3.411775 | +0.002267 | +0.131651 |
| 730237 | 3.264425 | 3.293393 | 3.403134 | +0.028967 | +0.138708 |

Positive gains favor generated latent states. All models see128,025,288 authentic
targets in the frozen order. Evaluation uses the same repeatedly inspected
384-window development panel and fixed60/20/20 domain weights.

New-seed comparison withordinary: mean gain0.015617, paired sampleSD0.018880; nominal two-sided95% t interval[-0.154016,0.185250]. Two seeds provide weak variance evidence; this is not fresh-source confirmation.
New-seed comparison withpause: mean gain0.135180, paired sampleSD0.004990; nominal two-sided95% t interval[0.090349,0.180011]. Two seeds provide weak variance evidence; this is not fresh-source confirmation.

The original selected seed had a0.025047-nat ordinary-control lead. The first
new seed has only0.002267nats, conditional source interval[-0.002717,0.007513];
the second has0.028967,[0.024391,0.033741]. The declared gate requires BOTH
new seeds to exceed0.01 with positive conditional intervals. It fails. Neither
the positive two-seed average nor the stronger second seed changes that rule.

The substantial advantage over trained constant pauses repeats in both seeds.
This supports a narrower equal-data finding about learned generated states,
not dependable superiority over the strongest ordinary predictor, general
reasoning, learned allocation, supported thought-length scaling or training
efficiency. The candidate still uses3.782219times the strongest ordinary
training matrix work. Arithmetic and typed-bracket diagnostics remain negative
or inconclusive.

The prepared ephemeral scientific comparison is NOT admitted under its original
replication gate. Its independently qualified implementation remains available,
but this report does not silently reroute the failed gate into a fit.

All six Native attempts total17.146444assigned chip-hours. Exact CPU readouts
use zero chips. A final aggregation path error occurred after both complete
readouts; the aggregate was rebuilt from their unchanged receipts without any
training or scoring rerun. The original failure remains preserved.

[Full result](report.json) · [Execution costs](execution-costs.json) ·
[Timing detail](../replication-runtime-v1/REPORT.md) ·
[Report recovery](../replication-report-recovery-v1/plan.json).
