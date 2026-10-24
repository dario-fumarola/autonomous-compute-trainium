# Status — 2026-10-24, 00:26 UTC

**All three mixture 1B endpoints are complete.** Fresh DEV NLL is
2.587427 / 2.590665 / 2.587213 for seeds103 / 211 / 307, after planned decay.
Their earlier100M/301M stable scores were3.808420/3.013947 and
3.814483/3.019806. Mixture103/211 and both ordinary103 endpoints have passed
independent whole-DEV FP32 audits and full optimizer-state copy/readback.
The two mixture CPU-minus-native deltas are-0.00001469 and-0.00000106 nats.
Five of twelve primary-model audits are complete, including all mixture endpoints.
[Partial audit](results/variable_depth_20261023/partial-independent-cpu-verification.json).
The first ordinary pair is also ready: at equal data, ordinary103 scores
2.599135 versus mixture103 at2.587427, a **0.011708-nat** gain. At matched
estimated training matmuls, ordinary at1.700B targets scores **2.495278**,
beating mixture at1.005B by **0.092150 nats**. This pair's equal-data gain
does not repay the extra compute. The remaining paired seeds and fixed-depth
controls are pending; all16 chips remain assigned.
[First-pair DEV comparison](results/variable_depth_20261023/first-pair-dev.json).
Ordinary211 now scores2.598596 at1B, versus mixture211 at2.590665: a
0.007931-nat equal-data gain in the second pair. Ordinary307 is pending.
No primary comparison or new TEST result has been reported.
The standalone [first-pair learning curve](results/variable_depth_20261023/first-pair-curve/first-pair-learning.png)
includes source-group uncertainty. At1B, seed103's conditional interval is
[0.009770,0.013632], which does not clear the0.01 useful-gain threshold.
Across all three paired seeds at301M, the mean gain is0.009592 with nominal
seed interval[-0.000256,0.019440]. These descriptive checks measure different
uncertainty and do not replace the frozen final analysis.
[Conditional readout](results/variable_depth_20261023/first-pair-curve/REPORT.md) ·
[Early seed variation](results/variable_depth_20261023/early-paired-curves.json).
At301M stable targets, ordinary103 scores3.024108 versus mixture103
3.013947: a0.010160-nat equal-data gain in one pair, with additional compute.
Mixture307 scores3.015378 after its worse100M point3.868176. These are
development curves, not a three-seed or matched-compute claim.
Selected peaks remain6e-4 ordinary,1e-3 fixed and6e-4 mixture, frozen after all
twelve300M-target trials. Fixed is an upper-grid winner, only0.002882 better
than6e-4; its optimum is not bracketed.
[Complete tuning](results/variable_depth_20261023/tuning.json) ·
[Live curves](results/variable_depth_20261023/progress.json).

The exploratory independent CPU cross-panel reference is complete. Old shared
recurrence still loses on fresh DEV: ordinary-minus-shared gain
**-0.031166 [-0.034496,-0.027835]** across the original three pairs. This rules
out a simple explanation based on changing the DEV panel; it does not isolate
which recipe change matters in the new study.
[Reference](results/variable_depth_20261023/old-recipe-fresh-dev.json).

The owner authorized ongoing work across days. This campaign still ends under
its frozen04:15 cutoff; the broader research goal remains active afterward.
A final positive needs a useful gain against a strong baseline and independent
confirmation, with repeated-testing accounting.
[Ongoing charter](docs/ONGOING_RESEARCH_CHARTER.md).
[Protocol](docs/VARIABLE_DEPTH_PROTOCOL_20261023.md) ·
[Frozen measured plan](results/variable_depth_20261023/plan.json).

All three paths have150,436,608 parameters. Ordinary, fixed-two-pass and the
one/two/three-pass mixture use the same explicit optimizer, data, routing,
global batch and per-arm four-LR selection budget. Mixed depths have mean two
and exactly match fixed-depth estimated neural training work. No pass signal,
separate norms, teacher, synthetic examples or RL is added.

The **77-test CPU suite passes**. Added tests cover operational recovery,
verified orphan-audit cleanup, the separate future
five-seed confirmation helper, source exclusions for a longer natural stream,
and a prospective topology resize; the running trainer, analysis and all30
frozen scientific source hashes are unchanged. No terminal confirmation
attempt has been registered. Native global-gradient relative errors are
0.00145 mixed /0.00155 fixed. Full-size CPU loss/gradient checks pass at each
depth, all replicas agree, optimizer state restores exactly and two fresh
continuations match bit-for-bit. Interrupted versus uninterrupted tiny training
is numerically close but not bit-identical (max parameter error3.31e-5,
global relativeL2 1.11e-6); this is explicitly separate from exact state
recovery. Normal-recipe AdamW CPU parameter error is2.98e-8.
[Qualification](results/variable_depth_20261023/qualification.json).

Over100 warm updates on eight chips/job, measured real targets/sec are
**900k ordinary /577k fixed /436k mixture**. The mixture has a real straggler
cost despite equal estimated work. The qualification-based primary budget is
5.39 node-hours including30% compute margin and an hour of setup/readout;
the longer one-seed diagnostic has a separate time-only launch gate.
The persistent queue and independent liveness/deadline monitor are running.
Training cutoff is October24 **03:00 UTC**, hard campaign deadline **04:15 UTC**.

All twelve completed LR trials also provide training-only launch-to-closure
rates of665k ordinary /452k fixed /351k mixture targets/sec per job. These
include setup/probe/checkpoint overhead but exclude separate DEV scoring;
they are serial job-time ratios, not summed node throughput.
[Timing receipts](results/variable_depth_20261023/tuning-timing.json).

A separate20-worker CPU job completed a **6,500,009,163-target natural
stream** for future longer runs. It rebuilt duplicate sketches for2,838,062
prior and fresh-panel documents. An independent full-array/SQLite audit
verified the entire old prefix, masks, vocabulary, checksums and zero exact-ID
or protected-group overlap. Preparation took30.88 minutes; the independent
audit took76 seconds. It changes no running data stream or model, and model-
source migration is not yet qualified.
[Data manifest](results/future_natural_extension_20261023/manifest.json) ·
[Independent audit](results/future_natural_extension_20261023/independent-audit.json).

A separate35-minute, at-most-eight-attempt topology qualification is prepared
to wait for successful scientific and read-only annex completion. It compares
32 ranks/batch4 with16 ranks/batch8 at the same source/depth assignments,
global batch, parameters and NN work. No native benchmark has launched yet.
It requires numerical/replica/recovery checks and100 warm updates per full
job; any speedup is a software result, not a learning result.

Fresh TUNE/DEV/TEST each retain at least250k natural targets/domain after
reconstructing all2,836,362 previous accepted documents' duplicate sketches
and excluding421,315 old source groups. The old TEST has no role in new
selection or confirmation. Raw data and all checkpoints remain private.
[Data gate](results/variable_depth_20261023/data-gate.json).

Prior fixed-depth results, protocols and evaluation hashes remain unchanged.
The earlier completed status follows as a historical record.

---

# Status — 2026-10-23, 19:50 UTC

**The bounded scientific campaign, independent audits, twelve-model sealed
TEST confirmation and post-science performance qualification are complete.
This recipe has no demonstrated compute-efficiency advantage. No training or
evaluation workers remain active; no larger experiment has launched.**

All checkpoint identities, trained depths, evaluator and analysis hashes were
frozen before TEST access and still match the released files. No model, LR,
endpoint or inference-depth choice uses TEST scores.
[Frozen confirmation](results/frozen_test_20261023/REPORT.md) ·
[Release](results/frozen_test_20261023/release.json).

At 1,004,887,590 real targets, the three-seed TEST means are:

| Model | Unique parameters | TEST NLL |
| --- | ---: | ---: |
| Ordinary | 150,436,608 | 2.578753 |
| Shared two-pass | 150,436,608 | 2.610307 |
| Untied two-pass | 251,124,480 | 2.549910 |

Ordinary-minus-shared gain is **-0.031554 [-0.038220, -0.024888]** across
training seeds. The declared 0.01-nat useful gain is ruled out at this exposure.
The original three-seed DEV negative remains preserved.

Untied improves over ordinary in every observed seed. Its combined TEST gain
is **0.028842 [0.002752, 0.054932]**: positive, but **inconclusive at the
declared 0.01-nat useful-gain threshold**. This interval includes discovery
seed23. The separate fresh-seed interval is much wider,
**[-0.093489, 0.155537]**. All seeds use the previously selected 3e-4 recipe
without new tuning. This is a larger-network equal-data advantage, with 67%
more parameters, rather than a parameter-matched or efficiency victory.
[Replication and preserved DEV readouts](results/confirmation_followup_20261023/REPORT.md).

The fully annealed ordinary resource control at 1,699,926,288 targets scores
**2.478950 TEST**, versus **2.550684 untied** at 1,004,887,590 targets in
seed23. Ordinary leads by **0.071734 [0.065560, 0.077985]**, conditional on
this fitted pair. Estimated dense training matmul budgets differ by 0.0133%;
the vocabulary projection is included. Recorded retained training paths take
42.40 minutes ordinary and 39.61 minutes untied23, with five versus two process
segments. The ordinary parent-prefix time is reconstructed from the original
checkpoint timestamp. This is not an exactly wall-time-matched comparison.
[Detailed resources](results/confirmation_followup_20261023/RESOURCES.md).

The unchanged shared seed23 continuation at 3,014,637,826 targets scores
**2.393539 TEST**, versus **2.391002 ordinary**. Ordinary-minus-shared gain
is **-0.002538 [-0.005603, 0.000730]**, conditional on one fitted pair.
The earlier deficit largely disappears, without establishing a useful gain.
The stable-checkpoint DEV advantage also disappears after endpoint annealing:
the annealed DEV gain is 0.000114. DEV and TEST are separate panels.
[Longer-exposure curve](results/sharing_exposure_20261023/REPORT.md) ·
[Figure](results/sharing_exposure_20261023/learning-curve-seed23.png).

All new independent full-panel FP32 audits pass; the largest new CPU/native
NLL difference is 0.00004176 nats. Earlier six-endpoint audits also passed.
Generic sharded evaluators reproduce scalar NLL exactly. Full optimizer,
RNG and data-cursor continuity and all-replica equality pass. Nine new
checksum-verified full-state copies join six earlier endpoints on existing
EBS. These are on-instance recovery copies, not off-instance backups. Raw
weights, optimizer state and source contents remain private. The root
filesystem retains approximately 39 GiB free, above the 35 GiB floor.

After scientific scoring closed, **unscaled fused AdamW qualification passed**
all distinct 150M weight shapes, offset gradient views, changing LRs, CPU
agreement and fresh-process full optimizer recovery. Two synchronized jobs on
disjoint eight-chip halves achieve **873,844 ordinary / 576,011 shared real
targets/sec**, each over 100 warm updates. The unequal-duration pair completes
52,337,076 targets in 45.43 seconds: **1,152,021 targets/sec across the pair**.
The individual rates must not be summed. These jointly change optimizer,
local batch and DDP topology; they do not isolate an optimizer speedup or
include a scientific checkpoint cadence. A separate exact-NLL readout check
takes 58.64 seconds with 16 ranks versus 85.90 with 64, one measurement each
with different cache histories. [Qualification](results/fast_path_20261023/REPORT.md).

A separate direct-ATen fixture reproduces ignored optional overflow/unscale
metadata in the pinned fused optimizer. Normal unscaled updates agree with
CPU. High-level GradScaler reachability remains untested; neither scientific
training nor the qualified unscaled candidate uses those optional arguments.
This does not explain the scientific NLL gaps. A reproducible contribution
note is ready for human review; no upstream submission has been made.
[OSS evidence](docs/OSS_FUSED_ADAMW_METADATA_20261023.md).

The incremental ledger closes **82 native attempts**, including stopped and
failed work: **90.60 assigned chip-hours**, **5.91 hours of accelerator-active
node time**, and **6.19 hours from first to last native launch**. The latter
includes CPU/control/idle gaps and corresponds to 99.03 node-opportunity
chip-hours; it is a lower bound on reservation cost, not the full instance
bill. Historical controls and prior campaigns are retained separately rather
than charged again. Known neural matmuls and numerical probes are itemized;
optimizer, communication and unknown failed work are not silently treated as
neural FLOPs. [Execution ledger](results/sharing_exposure_20261023/execution-costs.json).

Current CPU suite: **53 tests pass**. All scientific fits used all 16 chips
with 64 ranks. Natural web, mathematical web and repository code remain the
only research corpus: no generated examples, teacher, RL or learned halting.
Host drivers and XLA are preserved. All previous results and protocols remain
intact. Do not extend or scale this exact shared recipe on the evidence here.
Any selective-sharing experiment needs a separate protocol and held-out panel;
this TEST panel is now consumed.

[Mechanism diagnostic](results/recurrent_mechanism_20261023/REPORT.md) ·
[Original primary](results/recurrent_learning_20261023/REPORT.md) ·
[Follow-up protocol](docs/CONFIRMATION_FOLLOWUP_20261023.md) ·
[Prior status](docs/archive/STATUS_BEFORE_TEST_READOUT_20261023.md).
