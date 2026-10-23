# Variable-depth pretraining — frozen first campaign

## Question and falsifiable predictions

Does training a shared core at several depths make additional applications
useful for unseen natural-text prediction? Preserve the 150,436,608-parameter
pre-norm architecture: two prelude layers, sixteen shared core layers, two
coda layers, additive input injection, tied 32k vocabulary, context 1024 and
full backpropagation. No teacher, synthetic corpus, auxiliary loss, gates,
learned halting, pass-specific parameters or model scaling.

Compare ordinary (one pass), fixed (two passes) and mixture (one/two/three
passes). Read each recurrent endpoint at depths 1/2/3/4; depth four is declared
out-of-support extrapolation. Ordinary is read at one pass. The primary
mechanism contrast is mixture depth1 NLL minus depth3 NLL. The primary
architecture contrast is fixed depth2 minus mixture depth2 NLL. The practical
contrast is ordinary at matched estimated training matmul work minus mixture
depth2. These are separate claims. A mechanism win alone is not an efficiency
win, autonomous compute allocation or evidence of an interpretable algorithm.

This is motivated by the variable-depth training of
[Geiping et al.](https://arxiv.org/abs/2502.05171), not a reproduction or a
novel variable-depth architecture. Their large-model training differs in core
size, iteration distribution, normalization, truncation and source mixture.
We isolate training-depth support under the project's natural-source policy.

## Static distributed mixture

Use two disjoint eight-chip jobs when dependencies permit, each with 32 ranks,
local batch4, accumulation2: 256 source windows/global update. Ten ranks always
compile depth1, twelve depth2, ten depth3. Each global update therefore has
80/96/80 windows at depths1/2/3, probabilities 5/16, 6/16, 5/16 and mean depth
exactly two. This is a within-update mixture, not uniform 1/3 sampling or
whole-update locked-step sampling. DDP shares all parameters and averages the
real-target-normalized gradients from all three paths. Only final-path loss
is used for each example; common prefixes are not reused across examples.

All arms use the same deterministic within-update permutation of the original
256 windows, keyed by cursor and routing seed271828. The actual global set,
source order between updates, target masks and per-target coefficient are
identical across arms. Routing is independent of content, model weights and
outcomes. It prevents a fixed rank from receiving the same stream slots at
every update. Its explicit SplitMix64/Fisher-Yates algorithm is resumable and
independent of Python's RNG version; tiny modulo bias is irrelevant to data
selection and is disclosed. Real-target counts need not be exactly balanced
among depths because document-boundary masks vary. Every position is counted
in matmul accounting, including masked boundary targets.

This static-per-worker formulation avoids retaining three large compiled
graphs on every rank. It may have a depth3 straggler cost; measure it rather
than claiming the FLOP match implies a speed match. All controls in this
campaign rerun with identical routing, topology and unscaled fused AdamW,
FP32 master parameters/moments and BF16 matmuls. Prior plain-optimizer controls
are contextual references, not causal comparators. Scientific legacy code and
consumed TEST sources stay unchanged.

## Fresh held-out data

Reuse the frozen natural 60/20/20 packed stream and its append-only 3.25B
extension without wrapping. Build fresh TUNE, DEV and sealed TEST from unused
source groups in already downloaded, pinned raw shards. Exclude all source
groups and exact identities accepted in either previous corpus, including old
held-out panels. Reconstruct every previous accepted document's raw-text
near-duplicate sketch before accepting any new panel document; apply the same
approximate lexical duplicate screen against old and new examples. This is
not an exhaustive semantic duplicate guarantee.

Fresh group roles use a new hash salt, independent of model scores. Only
original held-out groups are eligible, with fixed content-hash subsampling.
Target at least 250k real targets per domain/role, preserving complete
documents and tails. TUNE and DEV are separated by group; TEST stays sealed
until checkpoint identities, analysis, contrasts and evaluator hashes freeze.
The old TEST panel is never used to develop or confirm this intervention.
Public artifacts contain hashes, provenance and aggregate counts only.

## Prospective common-recipe amendment

Before any new LR trial or scientific fit, make the new recipe explicit:
AdamW beta1=0.9, beta2=0.95, eps=1e-8, matrix weight decay0.1, zero decay on
all one-dimensional parameters and the tied embedding/head. Use the same
parameter-group construction in all arms. Tuning has2.5% token warmup; primary
fits have2.5% token warmup. The earlier default-beta2 qualification remains a
software control, not a scientific fit. Requalify the explicit recipe with
CPU/native AdamW parity, global gradients, full optimizer recovery and100 warm
updates before measured budgeting. This amendment is motivated by inspecting
the old optimizer defaults, not by new model outcomes. It does not establish
that the old recipe caused the recurrence loss. No pass embedding, separate
norms, adapters, altered injection or truncated gradient is added.

## Qualification and tuning gates

CPU tests must verify exact global-window coverage, deterministic resume,
unique-parameter equality, exact fixed/mixture dense-matmul equality and DDP
loss/gradient normalization with unequal target masks. Native qualification
must validate all three depth losses/full gradients against CPU, tiny global
mixed-depth gradients, numerical interrupted-versus-uninterrupted replay,
full-size optimizer recovery, identical replicas and zero unexpected fallbacks. The known `aten::_to_copy`
host-transfer fallback is allowed and reported; no math/optimizer fallback is
allowed. The first software attempt stopped because this expected reporting
copy was initially rejected; its updates and cost are retained.
Require CPU/native loss difference <=0.01 nats and full-gradient relative L2
<=0.01. Benchmark at least 100 warm updates of each recurrent path before
freezing the measured campaign budget. Require exact model/optimizer loading
and exact agreement between two fresh-process continuations from the same
checkpoint. Bit equality to an uninterrupted run is a stronger, separate
diagnostic: the legacy-beta2 tiny test showed maximum parameter drift7.09e-6
after8 further updates, while two fresh replays were identical. Its cause is
not isolated. Before testing the new recipe, set numerical replay limits:
maximum parameter difference5e-5, global parameter relativeL2<=1e-4,
first-moment maximum difference1e-5, second-moment maximum difference1e-7,
and exact step counters, cursor and exposure. Record actual errors and retain
failed exact-replay attempts. All qualification updates are discarded.

Each arm receives four 1152-update (301M-target) trials, tuning seed11,
29-update warmup, final115-update cosine decay and floor0.1. All three arms
use the same LR grid: 1.5e-4/3e-4/6e-4/1e-3. Choose minimum fresh
TUNE weighted NLL at depth1 ordinary, depth2 fixed/mixture. Preserve every
trial. Both recurrent selected trials must score below5.0 before primary
fits; a failed learning gate stops this experiment rather than adding trials.
A boundary optimum remains a stated limitation. No depth/LR selection uses DEV
or TEST. Freeze the selections before primary fits.

## Exposure, matched-compute endpoints and schedule

Fresh paired initialization seeds103/211/307; common source-order seed23.
Every arm must have the same initial model digest within seed. Preserve all
Adam moments, RNG, cursor, counters and LR history on resume. Main warmup96
updates, stable checkpoints384/1152/3456, then384-update decay to3840
(1,004,887,590 targets). The ordinary matched-training-matmul endpoint uses
6496 updates (1,699,926,288 targets), stable5846 then650-update decay. Branch
from the untouched ordinary3456 state, never from its annealed1B endpoint.
Mixture and fixed have exactly equal estimated training matmul work per update;
ordinary matched budget is 0.0133% smaller due to integer updates. Include all
vocabulary projections, every pass and full backward; elementwise/optimizer,
communication, compiler overhead and wall time are separate costs.

Reserve an additional seed103 three-arm 3B diagnostic from untouched3456
states: stable10368, decay1152 to11520 (3,014,637,826 targets). Execute only
if the qualification-based remaining-time estimate, with 30% margin, fits
before the fixed training cutoff; otherwise report unexecuted. This condition
uses time/resources, never DEV/TEST gains. No larger exposure or extra seed is
added based on outcomes. Stable learning-curve points and annealed branches
must be labeled separately.

## Readout and decisions

Primary report: fresh DEV and then one frozen TEST release, all three paired
seeds, per-domain and fixed60/20/20 weighted real-target NLL. Inference depth2
is the predeclared architecture/compute comparator, not the best observed
exit. Report each seed and depth, the 1-to-3 and 2-to-3 mechanism effects,
depth4 extrapolation and inference compute. For each declared three-seed
contrast use the existing paired Student-t95% interval: useful gain supported
if lower bound>0.01 nats, ruled out at this exposure if upper bound<0.01,
otherwise inconclusive. With three primary contrasts, also report a
Bonferroni-adjusted family interval; no global success claim based on selecting
one favorable contrast. Source-group intervals are separate and conditional
on fitted weights. A one-seed3B result cannot establish training-seed robustness.

Independently score the primary depth2 mixture/fixed and ordinary endpoints
with the separate FP32 CPU model; verify complete real-target coverage and
checkpoint hashes. Preserve full training states and restore-test before
reporting completion. No adaptation on TEST; new interventions need new panels.

## Bound and execution

Hard campaign deadline: **2026-10-24 04:15 UTC**. Scientific training cutoff:
**03:00 UTC**, reserving final readout/report time. Absolute cap8 node-hours
from20:15 UTC; do not allocate another instance or alter host drivers/XLA.
Keep512 GiB host MemAvailable and35 GiB free on every output filesystem.
Scientific state stays private on existing storage; never delete old results.
Keep failed/stopped/tuning/compilation/evaluation work in the ledger, and
separate assigned-chip time, active whole-node time and reservation span.
Persistent controllers must advance to every approved stage or leave an
explicit failed/unexecuted prerequisite. Final report must distinguish software
qualification, learning, prediction benefit and resource efficiency.
