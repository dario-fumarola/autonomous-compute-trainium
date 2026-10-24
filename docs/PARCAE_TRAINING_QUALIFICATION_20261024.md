# Low-depth Parcae training and throughput qualification

Identity `parcae-low-depth-qualification-v1`. Numerical/performance qualification,
not a scientific result, tuning selection or an automatic large-run launch.
Preserve the active common-FP32 comparison and all existing frozen gates.

Owner steering replaces the unlaunched eight-pass training benchmark. Its CPU
sampler/initializer preparation remains preserved; no eight-pass native training
or scientific fit was launched. Existing four-forward/two-backward numerical
gradient gates remain useful hardware coverage and finish unchanged.

Use the unperturbed initializer from clean upstream release69284c1, seed1009,
32k vocabulary, width768/heads6/FFN3072 and2/2/2 blocks. Ordinary has140,781,888
parameters; true one-pass and low-depth Parcae each have141,963,840. Every shared
backbone/value-embedding/head tensor is cloned across all arms. The two Parcae
arms share their entire initial weights. Ordinary omits injection/readout;
its comparison is not parameter-exact. Vocabulary and1024-token context are
explicit adaptations; positional capacity remains2048.

Compare ordinary, true constant depth one, and per-sequence Poisson(lambda2),
minimum one, backpropagation through the last application. The clamped expected
depth is2+exp(-2)=2.135335, not exactly two. Do not clamp the
maximum sampled forward depth. Prepare rank-specific, one-based-microstep
tables on CPU and pin their hashes. Check actual draws against the released
`_sample_batch_depths` method, including rank seeding. This keeps the sampler
reproducible across Torch runtimes. Initial states use the released truncated
normal law and an explicit CPU generator keyed by absolute source-window index.
These states are internal noise, never a source/target corpus.

Context1024,32ranks per eight-chip job, localbatch1, accumulation8 and16 give
256/512 windows per update. Natural source seed23, routing271828, no wrapping.
All arms consume the same real targets/order within geometry. Rank0 verifies
the entire source and mapped files before a barrier; every rank checks the same
frozen manifest before mapping. This removes redundant hashing, not validation.

The native queue requires a passing full FP32-gradient and native optimizer/
recovery gate, a terminal FP16-gradient gate and no managed native owners.
Always measure the qualified FP32 path. Add FP16-autocast body/FP32-head only
if its prior native gradient gate passed; otherwise keep FP32 alone and record
the numerical branch. No choice uses held-out losses or measured speed.

Measure all three arms at accumulation16; also measure the two recurrent-family
arms at accumulation8 to quantify the accumulation tradeoff. These are engine
profiles, not scientific depth/size sweeps. The single gradient-rule follow-up
will keep exactly the same Poisson forward depths and backpropagate through all
applications; it has no scientific launch in this qualification.

Use fused unscaled AdamW and the qualified release-policy Muon optimizer.
FP32 masters/moments/gradients, manual scale8192 for FP16, finite full-global
clipping at1 after no_sync accumulation/reduction. Conservative qualification
LR rises to1e-4 over20 updates; Muon momentum rises0.85→0.95 over20. This is
a throughput fixture, not the paper's intended scientific LR schedule.

Each primary's actual first update also gets an independent CPU clipping/
released-optimizer check using its exact observed raw global gradient. This
covers the zero-gradient matrices at the unperturbed initializer, which the
earlier nonzero prescribed-gradient fixtures do not. Apply2% update/moment gates
overall and per optimizer family; preserve the first-state/raw-vector hashes.
This arithmetic check does not replace the independent full-gradient check.

Each primary performs20 cold/warmup updates, **100 measured warm updates**,
then saves complete state after update120. Exclude initial probes/compilation,
final checkpoint, and numerical/replay audits from warm throughput; keep all
of those in campaign receipts. Warm timing includes source/H2D, state generation,
all actual recurrent work, communication, finite checks, clipping, optimizer,
and declared logging. Count every repeated Muon matrix update on all32 ranks
as well as head-inclusive model work. Separately report non-matmul costs/time.

An uninterrupted121st update records the complete unscaled gradient before
clipping. A fresh process restores update120 and repeats121. Require exact
model/optimizer and per-rank CPU RNG state on load, identical replica states,
exact source cursors/exposure/work counters, finite complete states, unchanged
persistent buffers and tied head values. Require resumed/uninterrupted update
and named-moment errors≤2%, overall and separately for each optimizer family.
Do not claim bit-identical future trajectories.

The cold local CPU/backend check uses the verified FP32 port on the same local
Torch runtime and **the exact supplied hidden state**, avoiding RNG-version
differences. The decisive updated-point check uses independently loaded pinned
upstream CPU FP32 code over all256/512 next-update natural TRAIN windows.
Save exact native-runtime states/depths privately for that reference. Both
primary and resumed full gradients must match within2% overall and per named
family, and NLL within0.003. Initial checks alone cannot pass the campaign.

Use both eight-chip halves for predeclared work with ownership-scoped cleanup.
Latest native phase start04:00UTC October25; each phase≤30 minutes, CPU audit
≤30 minutes, full execution/capture/auditing≤90 minutes. Two CPU audits may
overlap native jobs.160GiB/half and320GiB aggregate,512GiB MemAvailable and35GiB
disk reserves, existing SDK/drivers/XLA. Waiting assigns no chips. Include
failed/stopped paths and intentional source reuse across benchmark settings.

No held-out panel, terminal alpha or quality claim. Passing provides a measured
training engine and budget inputs. A scientific LR screen/comparison still
requires its own frozen data/compute endpoints, schedule, budget and controls.
