# Status — 2026-10-24, 15:42 UTC

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

## Completed follow-ups and next bounded qualification

The **5.100B ordinary compute control is complete and independently audited**:
ordinary2.342369879 versus fixed3B2.369851103. Ordinary wins by0.027481223 nats.
Both endpoints completed decay; total estimated training matmuls match within
0.003051%. Differing precision-history fractions are not matched chip-time costs.
This one-pair consumed-DEV efficiency result is negative.
[Complete resource comparison](results/mature_fp32_compute_20261024/REPORT.md).

The **trained inexpensive-connection pilot is complete**. At1.005B targets,
ordinary NLL is2.606832753 versus connection2.574477293, a0.032355460-nat gain.
Parameters, original weights, source order and estimated matmuls match. Both
train in FP32 from random initialization, each with its own four-LR301M screen.
All selected TUNE and all six main whole-DEV/state-copy audits pass. This is one
predeclared seed419; the gain is descriptive, not seed significance or a latent-
depth success. It supplies a stronger architectural baseline for recurrence.
[Complete connection report](results/reinjection_pilot_20261024/REPORT.md).

The review correctly identified shared-container failure cleanup as an execution
risk. Both scientific campaigns finished without that interruption. Historical
frozen sources remain unchanged. New launches use separate ownership-checked
containers: ordinary failures stop their own workers; shared host-memory/disk
emergencies can stop managed workers globally. A real CPU-worker test verified
that cancellation preserves the other container and its existing worker.
[Isolation proof](results/native_execution_isolation_20261024/report.json).

A new30-minute, zero-update native FP16 full-gradient qualification is registered
for all16 chips after the scientific native readouts closed. It uses an exact
local copy of the existing SDK root filesystem, two160GiB caps (320GiB total),
unchanged host drivers/XLA, FP32 master gradients, an FP32 vocabulary head and
manual loss scale8192. Compare both complete256-window mature objectives with
pinned CPU FP32 references. Numerical gates remain2% gradient /0.003 NLL.
No optimizer, recovery, throughput or learning adoption is implied by this test.
[Qualification protocol](docs/FP16_NATIVE_GRADIENT_PROTOCOL_20261024.md) ·
[Prepared plan](results/fp16_native_gradient_20261024/plan.json).

The CPU feasibility screen passed0.7251% scaled-gradient error; unscaled FP16
failed at8.86%. All active scientific comparisons kept their frozen FP32 path.
A retrospective whole-document oracle gives only0.000651 nats improvement for
mixture depth selection and still loses to matched-work ordinary in every seed.
That deprioritizes document-level routing for those existing checkpoints; it
says nothing universal about token routing or new architectures.
[CPU precision report](results/cpu_fp16_precision_screen_20261024/REPORT.md) ·
[Oracle report](results/document_depth_oracle_20261024/REPORT.md).

The research goal stays active. Fresh confirmation panels remain unscored, no
terminal alpha is spent, and no latent-computation efficiency win is claimed.
The next scientific comparison must retain the inexpensive connection as a
strong control and distinguish its gain from the value of another core pass.

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

The 101-test CPU suite passes after the latest source changes. Natural sources
only; no synthetic research corpus, external teacher, RL or larger model.
All failed/stopped jobs, tuning, qualification, compilation, data preparation
and CPU tails retain separate receipts. Concurrent intervals must not be added
as sequential reservation time. Existing old results/checkpoints are preserved.

[Previous full status and historical links](docs/archive/STATUS_BEFORE_FP32_3B_COMPLETION_20261024.md) ·
[Ongoing research charter](docs/ONGOING_RESEARCH_CHARTER.md).

An unapplied local SDPA dtype-policy contribution draft now has seven CPU gate fixtures and clean patch-application checks. It was neither installed nor submitted upstream. Patched native/cache/performance validation remains pending. [Draft](contributions/sdpa_dtype_policy/README.md).
