# Status — 2026-10-24, 09:14 UTC

**The paired 3B FP32 continuation is complete and independently audited.** At
3,014,637,826 real targets, ordinary DEV NLL is **2.405907** and fixed two-pass
NLL is **2.369851**: a **0.036056-nat equal-data gain**. Both models have
150,436,608 parameters and completed the registered learning-rate decay.

| Domain | Ordinary NLL | Fixed NLL | Gain |
| --- | ---: | ---: | ---: |
| Web | 2.879228 | 2.838200 | +0.041028 |
| Math | 2.091221 | 2.056372 | +0.034849 |
| Code | 1.300630 | 1.278284 | +0.022345 |

This is one reused paired seed103 on consumed DEV, with BF16 history below
2.010B and qualified FP32 thereafter. It is a descriptive longer learning
curve, not fresh significance, a pure timing effect or an efficiency win.
All six whole-DEV independent FP32 scores and full optimizer/model copy audits
pass. Maximum native/CPU NLL difference is 0.000000808; all 37 frozen scientific
source hashes remain unchanged. Copies are on this instance.

[Completed report](results/mature_fp32_20261024/REPORT.md) ·
[Exact summary](results/mature_fp32_20261024/summary.json) ·
[Standalone learning curve](results/mature_fp32_20261024/figures/paired-learning-curve.png)
([SVG](results/mature_fp32_20261024/figures/paired-learning-curve.svg),
[PDF](results/mature_fp32_20261024/figures/paired-learning-curve.pdf)).

## Active work on the instance

**Half0, eight chips:** the missing ordinary 5.100B resource control is training
from the independently audited **stable 2.713B** checkpoint. It retains the
qualified FP32 path and completes its own 4.590B→5.100B decay. Its estimated
training matmul budget matches fixed3B within 0.003051%, including the vocabulary
head. This is the still-pending efficiency comparison. The registered measured
window is 5.60 hours including 30% margin and CPU tail; training cutoff14:45 UTC,
hard cutoff15:30 UTC. No numerical threshold changed.

[Control protocol](docs/MATURE_FP32_COMPUTE_CONTROL_20261024.md) ·
[Frozen plan](results/mature_fp32_compute_20261024/plan.json) ·
[Live observations](results/mature_fp32_compute_20261024/live-observation.json).

**Half1, eight chips:** a separately registered pilot is starting its four-LR
selection for ordinary versus an inexpensive trained reinjection connection.
Each model uses every block once; the control adds the prelude state at the
core boundary. Parameters and estimated matmuls match ordinary. This tests a
missing architectural control, not a new latent-depth success claim.

Native qualification passed actual updates, a 32-window global-gradient check
(relative error 0.000000708), 100 warm updates at the intended 256-window batch
(**176,832 targets/s**), and eight full-state restored updates. All replicas agree.
The whole native sequence took 15.35 minutes, within its 25-minute bound.

Both pilot arms receive LRs {0.0003,0.0006,0.001,0.0015}, 301M-target tuning,
seed11, matched weights/source order, FP32 and the same AdamW recipe. Selected
TUNE models receive independent CPU audits. A predeclared >0.02-nat cheap-model
TUNE penalty stops the main fits; otherwise both train seed419 to1.005B with
stable learning-curve points and completed decay. Half0 can take remaining
predeclared tuning trials after the strong resource control closes. The
conservative measured pilot budget is9.10 hours including margin/tails; training
cutoff18:00 UTC and hard cutoff19:00 UTC. Existing consumed TUNE/DEV only.

[Pilot protocol](docs/REINJECTION_PILOT_PROTOCOL_20261024.md) ·
[Measured frozen pilot plan](results/reinjection_pilot_20261024/plan.json) ·
[Live observations](results/reinjection_pilot_20261024/live-observation.json) ·
[Completed native qualification](results/reinjection_control_20261024/native-v1/report.json).

No fresh TEST is opened and no terminal alpha is spent. A future positive still
requires a frozen candidate and strong controls, at least five fresh paired
training seeds, new held-out groups and the charter's repeated-test allocation.
The research objective remains active.

## The established three-seed result

The earlier 1B experiment and sealed TEST analysis remain unchanged:

| Model | Targets | Relative estimated matmuls | Mean TEST NLL |
| --- | ---: | ---: | ---: |
| Ordinary | 1.005B | 1.000 | 2.587896 |
| Fixed two passes | 1.005B | 1.692 | 2.535153 |
| Variable-depth mixture at two passes | 1.005B | 1.692 | 2.575693 |
| Ordinary, matched estimated work | 1.700B | 1.692 | 2.483895 |

Fixed's equal-data gain is0.052743 nats, with nominal paired-seed95% interval
[0.041749,0.063737], positive in all three seeds. Its compute-matched deficit is
0.051258. Variable depth loses to fixed at depth two; supported-depth dependence
does not establish monotonic refinement or learned halting. The original TEST
panel is consumed. [Frozen TEST report](results/variable_depth_20261023/test/REPORT.md).

## Qualified fresh data and preserved numerical findings

The new natural-source confirmation reserve is complete: 1,697 documents and
2,261,216 targets across TUNE/DEV/TEST. All14,700,735 prior accepted IDs were
screened. Independent raw-row/full-token reconstruction, coverage, exact-ID and
source-group checks pass. No model has scored these panels. The auditor did not
independently rerun every approximate minhash comparison; that limitation remains
explicit. The memory repair reuses the exact frozen candidate pool and leaves
both stopped predecessors and costs preserved.

[Fresh-reserve report](results/fresh_confirmation_reserve_20261024/v3/REPORT.md).

The earlier BF16 mature run stays stopped at its unchanged numerical guards.
The full-model gradient controls distinguish BF16 error from the native fast
FP32 SDPA precision gap. Disabling that attention route reduces the fixed
full-global FP32 gradient error from0.029312 to0.00012068. The public checkpoint-free
reproducer and upstream contribution notes are preserved; nothing was sent
upstream. The failed strict future-trajectory bit test remains failed; the
prospective numerical policy promises exact state on load and qualified
arithmetic, not bit-identical future process-restart trajectories.

[Numerical diagnosis](results/trained_gradient_diagnosis_20261024/REPORT.md) ·
[Numerical recovery policy](docs/FP32_NUMERICAL_QUALIFICATION_20261024.md) ·
[OSS notes](docs/OSS_NOTES.md).

The historical fixed2B scoring audit stays failed at−0.000201372 nats against
its old BF16 readout. Its independent FP32 remeasurement supplies a common-precision
parent anchor without waiving that failure. An FP32-head/BF16-body CPU gradient
screen reduced error to3.37%, still above2%; no native adoption followed.
The first cheap-control CPU checker also failed: it packed reference Q/K/V
gradients incorrectly. Its repaired, independently checked fixture gives5.13e-7
relative gradient error and zero loss difference, with the original failure and
a regression test preserved. These are numerical diagnostics, not learning gains.

The 96-test CPU suite passes after the latest source changes. Natural sources
only; no synthetic research corpus, external teacher, RL or larger model.
All failed/stopped jobs, tuning, qualification, compilation, data preparation
and CPU tails retain separate receipts. Concurrent intervals must not be added
as sequential reservation time. Existing old results/checkpoints are preserved.

[Previous full status and historical links](docs/archive/STATUS_BEFORE_FP32_3B_COMPLETION_20261024.md) ·
[Ongoing research charter](docs/ONGOING_RESEARCH_CHARTER.md).
