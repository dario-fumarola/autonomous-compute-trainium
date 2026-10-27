# Learning the content and value of intermediate work

The owner explicitly replaces the stopped recurrence program with this new
research objective: learn useful self-generated workspaces from ordinary text,
then learn when that work earns its cost. Preserve all earlier findings;
none becomes positive under this change. The old goal remains unachieved.

The eventual question is random-initialization pretraining with early-active
versus delayed-onset workspace learning, including every common-prefix cost.
First use a provenance-pinned pretrained model to qualify and discover the
mechanism. A continuation result does not answer the from-scratch question.
No external expert teacher, supplied thoughts or generated source corpus.
Online temporary sampled thoughts and policy gradients are now authorized.

## First implementation and qualification

Use one actor for text and workspace generation, and an independently frozen
lagged copy for BOTH with-workspace and no-workspace scores. The actor never
receives future targets while generating or choosing an evaluation workspace.
The observed continuation is a TRAIN reward label only. Reader NTP learns from
generated workspaces plus natural targets alongside plain natural-text NTP.
Policy gradients use detached leave-one-out group advantages. Log absolute
gain too: the least harmful member of an all-harmful group is not a success.

Start the implementation with G=4 independent samples, workspace lengths
8/16/32, a four-token reward horizon and one uniformly sampled anchor per
256 eligible source positions. These are qualification settings, not an
admitted scientific budget or presumed2–3x overhead. Use the existing60/20/20
web/math/code source mix, disjoint source groups and explicit new-tokenizer
views. Freeze actual anchors, tokenizer and source hashes before scoring.
Never retokenize by silently treating old token IDs as new ones.

First establish reader sensitivity at fixed weights. Compare empty framing,
sampled work, fixed-token pauses and independent/past-only donor work. A
privileged true-future copy may be a separately labeled software positive
control only; never a generated workspace, research result, policy input or
inference choice. Embedding-gradient magnitude demonstrates dependence, not
useful reasoning. A randomly initialized reader need not have exactly zero
sensitivity. Diagnose signal and variance rather than assuming either a
capability floor or automatic useful reward.

Qualify on CPU before Native porting: exact target alignment, prefix-only
sampling, detached rewards, frozen scorer, whole-trajectory policy gradient,
real reader gradients, empty/variable lengths, deterministic RNG replay and
recomputation after content edits. The first reference uses uncached full
recomputation deliberately; its expensive timing is not a deployment estimate.
No optimizer campaign or fresh confirmation data is admitted by unit tests.

## Discovery comparison after qualification and measured budget

Compare ordinary continuation, workspace policy+reader learning, a trained
fixed-token extra-computation control, and a draft-continuation control. Give
the draft writer natural continuation labels, then let its reader use its OWN
sampled draft; never substitute the gold draft at inference. Give all conditions
comparable tuning and count every readout, generation, scorer, baseline,
forward/backward, optimizer and replay operation. Report source exposure,
repeated labels and generated tokens separately. At fixed allowance the cost
penalty is constant within a group and cannot teach allocation. Do not claim
halting before a later variable-budget controller is trained and tested.

The lagged no-workspace scalar cancels from centered group advantages. Keep it
for interpreting absolute utility; do not pretend subtraction creates another
learning signal. Start with an on-policy update, avoiding an unnecessary PPO
replay/ratio mechanism until it has a specific purpose. Lagged-reader updates
occur only after an actor update and are separately recorded.

Before scientific training, freeze the total budget, exact coefficients,
supported lengths, source mixture, CPU/Native numerical gates, full recovery,
measured cost and primary comparison. Do not choose fit duration from favorable
readouts. Human-looking thought text is not success and is not published with
private natural prefixes. No full from-scratch campaign follows automatically.

## Evaluation and claim hierarchy

Use a fixed random sample of held-out anchors, not sequential generation at
every position of the old750k-target panel. Broad loss, frozen reference-loss
slices and natural reasoning targets remain distinct. Use continuous answer
NLL and first-answer-token NLL where accuracy has a floor; report generation
accuracy secondarily. Reference difficulty is not a label for reasoning.

Test predictive usefulness, content-sensitive use, transferable capability,
supported inference budgets, then learned allocation as separate claims.
Measure absolute quality as well as within-model thought gains so a degraded
no-thought path cannot create apparent development. Compare each checkpoint
against the independently trained ordinary condition at matched resources.

For content interventions recompute ALL thought-associated states and K/V;
never edit visible tokens but leave original cached information. Preserve
length, positions and processing allowance. Use independent or past-only
donors to avoid answer leakage. Corruption alone is not proof of useful
reasoning. Targeted quantity edits are conditional on legible workspaces;
shorthand still permits whole-content controls but weaker interpretation.
Compare the draft arm before calling lookahead a distinct reasoning mechanism.

Use lengths0/8/16/32 only with declared training support and uncertainty.
Improvement across a useful finite range matters; strict per-example monotonic
scaling and unlimited extrapolation are not definitions of thought. Deployment
pays sequential workspace generation at every selected position; prefill timing
is insufficient. Allocation later compares learned, random, fixed and confidence
policies at the same ACTUAL cost, including the selection signal.

The proposed0.03 hard-slice gain and0.005 broad-loss tolerance are candidate
practical thresholds, not inherited positives or universal constants. Freeze
confirmation rules and adequately powered fresh-seed/source tests after the
pilot and before confirmation. Discovery may find a useful mechanism that
loses on efficiency; report both. A null applies to the tested model, data,
budget and recipe, not all self-generated computation.

## Provenance and prior work

Initial discovery candidate: allenai/OLMo-2-0425-1B, stage1-step1907359-tokens4001B,
immutable commit9d3e43659f00c17e6da23cf32333afd1fc39fa1a. Actual parameter
inventory1,484,916,736, not exactly1B. Avoid the final main/stage-two checkpoint,
whose documented mixture includes instruction and synthetic mathematical data.
Stage one has documented natural web/scientific/code/math provenance; this is
not an independent audit of every upstream pretraining document.
Source-group isolation in our continuation also cannot guarantee that the
upstream base never saw an evaluation document. Report that limitation;
relative continuation comparisons do not prove novel-task acquisition from zero.

[Model card](https://huggingface.co/allenai/OLMo-2-0425-1B) ·
[Stage-one files](https://huggingface.co/allenai/OLMo-2-0425-1B/tree/9d3e43659f00c17e6da23cf32333afd1fc39fa1a) ·
[Stage-one mixture](https://huggingface.co/datasets/allenai/olmo-mix-1124).

This is a Quiet-STaR/RLP-inspired hybrid, not an exact reproduction. RLP's
algorithm updates thought tokens without an ordinary NTP term; our reader+NTP
term is a proposed variant. Sparse anchors already appear in that literature.
PonderLM is a legitimate continuous alternative, not a lesser definition of
thinking. None establishes a universal1–2B lower capability limit.

[Quiet-STaR](https://arxiv.org/html/2403.09629v2) ·
[RLP](https://arxiv.org/html/2510.01265v2) ·
[PonderLM](https://arxiv.org/html/2505.20674v3).
