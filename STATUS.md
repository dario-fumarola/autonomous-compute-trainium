# Status — 2026-10-23, 22:10 UTC

**All twelve300M-target LR trials are complete, and the three-seed main fits
have started on both eight-chip halves.** Selected peaks are6e-4 ordinary,
1e-3 fixed and6e-4 mixture. Their TUNE NLLs are2.969665 /2.940222 /2.957476.
Ordinary and mixture winners are interior; fixed is an upper-grid winner,
only0.002882 better than6e-4. The optimum is not bracketed for that arm.
These are development selections, not architecture results.
[Complete tuning table](results/variable_depth_20261023/tuning.json) ·
[Live receipts](results/variable_depth_20261023/progress.json).
[Protocol](docs/VARIABLE_DEPTH_PROTOCOL_20261023.md) ·
[Frozen measured plan](results/variable_depth_20261023/plan.json).

All three paths have150,436,608 parameters. Ordinary, fixed-two-pass and the
one/two/three-pass mixture use the same explicit optimizer, data, routing,
global batch and per-arm four-LR selection budget. Mixed depths have mean two
and exactly match fixed-depth estimated neural training work. No pass signal,
separate norms, teacher, synthetic examples or RL is added.

The **60-test CPU suite passes**. Native global-gradient relative errors are
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
