# Recovery-only execution passes

Two32-rank/eight-chip halves load the retained passing FP32-polar primary
checkpoints and replay only the prescribed third update. Complete state on load
is exact, every replica agrees, and resumed versus uninterrupted updates have
relative error0 in both models. CPU comparisons remain below2% overall and in
each optimizer family. Third-step Muon errors are0.08674% ordinary and0.08207%
recurrent; named moments agree within1.15e-6 relative. Active92.135s.

No three-step primary was repeated. Count three retained primary calls and
one new replay call per replica per arm separately; all are numerical fixtures,
zero scientific updates or held-out access. The failed preceding replay's
ordinary call count remains unknown0–1 if no terminal artifact proves it.
Do not infer bit-identical future full-training trajectories from pure optimizer
replay. Both128GiB owned worker caps and the512GiB reserve stay enforced.

[Exact comparisons](native-report.json) · [Frozen inputs/source](native-plan.json) ·
[New execution costs](execution-costs.json) ·
[Original primary/failed replay costs](../native/native-execution-costs.json).
