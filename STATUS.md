# Status — 2026-10-24, 03:15 UTC

**The three-seed study, sealed TEST evaluation and read-only diagnostics are complete.** Fixed recurrence improves prediction at equal text exposure. Ordinary training still wins at matched estimated compute. No efficiency success is claimed.

| Model | Real targets | Relative training matmuls | Mean TEST NLL |
| --- | ---: | ---: | ---: |
| Ordinary | 1.005B | 1.000 | 2.587896 |
| Fixed two passes | 1.005B | 1.692 | 2.535153 |
| Variable-depth mixture, evaluated at two passes | 1.005B | 1.692 | 2.575693 |
| Ordinary, matched estimated work | 1.700B | 1.692 | 2.483895 |

All models have150,436,608 parameters, matched initial weights and source order within each seed. Fixed's equal-data TEST gain is **0.052743 nats**, with nominal paired-seed95% interval **[0.041749,0.063737]**. This declared secondary contrast is positive in all three seeds. Its descriptive deficit against compute-matched ordinary is0.051258 nats. Estimated work includes the vocabulary projection; these are not wall-time-matched endpoints.

The original three primary TEST contrasts give:

| Contrast, positive favors candidate | Gain | Family95% interval | Decision at0.01 |
| --- | ---: | --- | --- |
| Mixture, one to three passes | +0.027214 | [0.024333,0.030094] | Useful within-model gain supported |
| Fixed two to mixture two | -0.040540 | [-0.057493,-0.023587] | Useful gain ruled out |
| Matched-work ordinary to mixture two | -0.091798 | [-0.101169,-0.082427] | Useful gain ruled out |

Mixture's second-to-third pass worsens TEST NLL by0.003378; third-to-fourth worsens it by0.010618. It learns a useful supported computation path, without monotonically improving at every depth. This is not learned halting, a timing-from-initialization comparison, or an efficiency win.

[Complete frozen TEST report](results/variable_depth_20261023/test/REPORT.md) · [DEV analysis](results/variable_depth_20261023/dev/REPORT.md) · [Standalone learning/depth figure](results/variable_depth_20261023/figures/learning-and-depth-test.png) ([SVG](results/variable_depth_20261023/figures/learning-and-depth-test.svg), [PDF](results/variable_depth_20261023/figures/learning-and-depth-test.pdf)).

All twelve canonical models pass independent whole-DEV FP32 evaluation and serialized full optimizer/model copy recovery; the largest CPU/native NLL difference is0.00004622. All six mixture mechanism cells at depths1/3 also pass independent whole-DEV FP32 audits. Thirty DEV and thirty TEST evaluations closed successfully. The frozen release preceded the first TEST launch; all thirty original scientific source hashes and checkpoint identities remain unchanged. Independent numeric scoring is on DEV, not a separate whole-TEST CPU reproduction. Copies are on this instance, not off-instance backups.

[Canonical independent audits](results/variable_depth_20261023/independent-cpu-verification.json) · [Mechanism audits](results/variable_depth_20261023/independent-mechanism-verification.json) · [Frozen-record audit](results/variable_depth_20261023/frozen-record-audit.json) · [Completed read-only annex](results/variable_depth_20261023/readonly-diagnostics.json) · [Resource paths](results/variable_depth_20261023/model-resources/REPORT.md).

The additional completed-decay301M reference is **2.961808 ordinary /2.933570 fixed /2.949921 mixture** on DEV. This uses the previously TUNE-selected checkpoints of one different seed,11. It differs from the stable-phase301M main curves and cannot validate a long-horizon scaling law. Selected peaks remain6e-4 ordinary,1e-3 fixed,6e-4 mixture. Fixed wins at the upper edge of the bounded grid; its optimum is not bracketed.
[Exploratory short-exposure reference](results/selected_tune_reference_20261024/report.json).

The optional3B continuation did not launch under the completed campaign's time gate. The broader research goal remains active. The next question is whether the healthy fixed recipe's equal-data advantage can repay its compute cost at a more mature exposure. No new longer scientific fit has launched. Future confirmation needs a newly registered protocol, fresh held-out groups, independently replicated seeds and the ongoing charter's repeated-testing accounting. Existing TEST is consumed.
[Ongoing charter](docs/ONGOING_RESEARCH_CHARTER.md) · [Original frozen protocol](docs/VARIABLE_DEPTH_PROTOCOL_20261023.md).

A separate CPU source-copy qualification preserves every model, optimizer-moment and RNG tensor from ordinary/fixed seed103 stable904M checkpoints while advancing their source fingerprint to the audited6.5B descendant. Counters and schedules remain unchanged. This is metadata/copy qualification only; native continued-update equivalence is now running on all16 chips in the established layout. Four bounded eight-update attempts compare old versus appended-source restores for ordinary and fixed paths. These repeated updates are qualification only.
[Copy qualification](results/source_state_copy_qualification_20261024/report.json).

The queued mixture topology benchmark was explicitly superseded before any native launch. Its fixed-depth replacement attempted two tiny distributed jobs. The established32-rank layout passed; the16-rank layout spread over noncontiguous cores failed at collective initialization (`no_hier no_mesh`), before any model updates. No speedup or resized scientific run is established. Both attempts and idle opportunity time are charged. The established eight-chip/32-rank layout remains the fallback; no driver/environment changes are planned.
[Bounded failed qualification](results/fixed_topology_qualification_20261024/report.json).

Natural-data preparation toward17B stopped honestly: the pinned eligible code source exhausted at2,001,261,346 new targets, short of the2.1B domain quota. Prior duplicate reconstruction and accepted records remain preserved. The failed attempt is not a qualified corpus. A separate capacity-balanced finalization will retain whole accepted-document prefixes at approximately60/20/20, producing roughly16.5B targets without new sources or relaxed filters. It requires independent prefix, provenance and subset audits. Existing independently audited6.5B data already suffices for a3B-versus-compute-control screening run.
[Failed preparation costs](results/future_natural_extension_20261024/execution-costs.json) · [Existing6.5B audit](results/future_natural_extension_20261023/independent-audit.json).

The **84-test CPU suite passes**. A [registered CPU boundary-injection diagnostic](results/injection_only_dev_20261024/REPORT.md) is complete: the second core improves these same-weight predictions beyond the cheap reinjection-only ablation, with the limitations of an untrained intervention. Natural sources only: no synthetic research data, external teacher, RL or larger model. Failed attempts, compilation, tuning, CPU tails and overlapping elapsed intervals remain in separate resource ledgers. The prior full status is [archived](docs/archive/STATUS_BEFORE_FINAL_VARIABLE_DEPTH_READOUT_20261024.md).
