# Low-depth natural-text pilot: scientific charter

Owner steering, October24: retain the research goal, replace a high-depth main
study with a small-core, low-depth comparison and strong one-pass controls.
This charter specifies the direction and budget ceiling. **No scientific
launcher or fit is authorized by this file alone**: actual throughput, engine
gates, final endpoint tables, tuning allowance and execution window must be
registered before launch. Existing frozen fits finish unchanged.

The new truncation study supports prioritizing few fully differentiated loops
when forward/backward counts are jointly chosen. It also finds truncation can
pay when forward count is fixed. Its concatenation/RMSNorm update, MuonH recipe,
non-embedding size convention and fixed-depth experiments differ from Parcae;
its token/parameter crossover is not a threshold for our embedding-heavy models.
[Primary paper, revised October20](https://arxiv.org/html/2610.26030v2).
Parcae supplies a relevant small-model architecture and within-family
compute-allocation reference, not a guarantee for our corpus/precision/budget.
[Parcae primary paper](https://arxiv.org/html/2604.12946v1).

## Primary contrast and controls

Use natural web/math/code, random initialization, next-token loss, no teacher,
RL, generated corpus, router, new prediction objective or size/depth sweep.
Keep width768,6heads, FFN3072,2prelude/2core/2coda and the existing32k tokenizer.
Preserve all sources/weights/results; qualification weights never start a fit.

| Arm | Compute | Unique parameters | Interpretation |
| --- | --- | ---: | --- |
| A | Ordinary6-block recipe, value embeddings, same optimizer family | 140,781,888 | New backbone/optimizer reference |
| B | True one-pass Parcae | 141,963,840 | Injection/normalization/readout machinery |
| C | Per-sequence Poisson(lambda2), clamped minimum1; last1 application differentiated | 141,963,840 | Additional recurrent computation |

Every shared backbone/head/value-embedding tensor is cloned at initialization.
B/C share their entire initial weights. A/B differ in machinery and parameter
count; report that explicitly. **C versus B** is the parameter-exact recurrence
contrast. Do not call A the best ordinary configuration. Keep the trained cheap
connection as a practical challenger; old seed419 scores are references, not a
paired causal control for a new seed. Confirmation must retrain the strongest
eligible one-pass control on the fresh paired seeds.

**Prospective optimizer precision amendment, before scientific launch:** both
compiled and eager BF16 polar executions fail the unchanged2% actual-update
gate (about10.4% and2.32% respectively). Qualify an explicit FP32-polar variant
against independent pinned-upstream code with just its polar dtype changed.
Every arm must use the same selected qualified variant, with actual cost
measured. This is a Parcae architecture/optimizer-family experiment with an
explicit numerical deviation, not an exact release-recipe reproduction. No
scientific run starts while this optimizer/recovery/profile gate is unresolved.
[Preserved BF16 evidence](../results/parcae_early_optimizer_20261024/REPORT.md) ·
[Eager evidence](../results/parcae_eager_optimizer_20261024/REPORT.md).

The Poisson parameter is2, so expected executed depth is2+exp(-2)=2.135335.
B is deterministically1, not a requested Poisson mean1. Charge actual forward
counts and backward tails, including rare depths above2. Train support includes
one and two passes. Freeze the primary C readout at2 passes before scoring.

## Finite budgets and schedules

Anchor each main arm to the current ordinary1.700B path's complete matrix work:
6,496 updates ×286,044,821,913,600 = **1,858,147,163,150,745,600 FLOPs**,
full-square attention/FMA2. The old path has no optimizer matrix multiplies.
Reconcile the new ledger by adding **all replicated Muon matrix work**, output
projection, backward, any padding/recomputation and other executed matrix work.
Pointwise/embedding/communication overhead and chip-hours remain separate.
Mixed requested dtypes make nominal operation matching distinct from chip time.

Use measured100-warm-update rates to freeze feasibility and a separate capped
qualification/tuning allowance. Choose a common global batch before quality
outcomes. Independently select LR for A/B/C under the same frozen candidate
grid and selection budget;300M-target screening is the provisional minimum.
No scientific run starts from these conservative1e-4 qualification schedules.
Reduce the common main budget before outcomes if the full allocation does not fit.

The pinned small-model release launch files use8e-3 for both AdamW and Muon,
not the generic optimizer helper's3e-4/2e-2 defaults. Preserve the1:1 rate ratio
when specifying the new family's common LR grid; select an arm-specific scalar
rate under equal opportunities. The exact grid, warmup, cautious-decay schedule
and completed cooldown are still to be frozen before the scientific screen.
The old20-block AdamW grid and the conservative qualification rate are not
automatically suitable scientific settings for this family.

Precompute exact endpoints from the frozen depth/source stream. Round to whole
global updates and disclose the realized work discrepancy. Complete each arm's
own cooldown. Also reserve a1.005B-real-target equal-data branch per arm with
its own cooldown from the untouched stable trunk. Charge extra branches to the
campaign. Never compare a stable checkpoint with an annealed endpoint, or
continue a decayed branch as though it were an unchanged trunk.

Start with one paired development seed on old TUNE/DEV. Freeze the scientific
seed and endpoint tables in the launch registration. The existing fresh reserve
stays unscored. Keep weighted60/20/20 real-token NLL and0.01-nat useful-effect
threshold. No pilot source bootstrap stands in for training-seed variation.
Terminal positive confirmation requires at least5 fresh paired seeds, new
source groups and the charter's alpha spending rule; no terminal attempt here.

## One gradient-rule follow-up

Reserve one conditional full-BPTT follow-up with **exactly the same sampled
forward depths**, initial weights, architecture, data order and evaluation as C.
Change only the differentiated tail from1 to the complete sampled depth.
Keep the selected C learning-rate recipe fixed for this causal screen and give
the follow-up its own matched-work endpoint/cooldown. Any separate retuning would
be a new recipe comparison and must be named/costed. Do not compare fixed2/full
against stochastic2/truncated and attribute the contrast solely to gradients.

Advance to that follow-up only for a credible prespecified supported-depth
response or useful equal-data gain that has not earned its additional compute.
If C improves neither the one-pass machinery nor supported exits at the declared
budget, stop it. If the bounded family and this follow-up both fail, close that
family at this budget. Do not automatically increase depth, model size or exposure.

## Natural-code context diagnostic, before model scoring

Mine authentic held-out Python source using syntax/positions only, without any
model scoring or rewriting. Select identifier uses with a unique earlier local
definition in the same function; exclude ambiguous shadowing/control-flow cases.
Freeze extraction, quotas, source groups and token-distance bands before scoring.
Targets remain the original identifier token spans. Definitions must fit the
1024-token full prefix. Use distance bands129–256,257–512,513–900 and a128-token
short suffix. Record all exclusions and tokenizer byte/character alignment.
Development data may be consumed DEV; independent confirmation uses new groups.

For each target, score the same checkpoint at supported exits1/2, each with the
full original prefix and the fixed short suffix. Match random hidden states on
overlapping positions; preserve causal masking and target alignment. Validate
that context removal does not silently alter an unrelated position/state policy.
Use only original targets with ordinary autoregressive conditioning within spans.

G_full = L(full,1) − L(full,2).
Interaction = G_full − [L(short,1) − L(short,2)].

Require both G_full>0 and interaction>0 for the context-use finding. Positive
interaction alone can result from harming short-context predictions. Report
repo-group uncertainty conditional on each model separately from seed variation.
This is evidence about use of distant natural context, including retrieval;
do not label it general reasoning. A later functional endpoint may use existing
human-written repository tests with fixed decoding/sample budgets, never tests
generated for this study. It is not part of the immediate launch.

## Decisions

If only A/B improve, keep the stronger baseline and close recurrence at this
budget. A C/B win that loses to the stronger one-pass challenger remains a
within-family result. A compute-NLL win without the diagnostic supports only
prediction efficiency. A useful compute gain plus supported context benefit can
advance to fresh confirmation. Otherwise report negative or inconclusive under
the original threshold; do not move weights, exclusions or success criteria.
