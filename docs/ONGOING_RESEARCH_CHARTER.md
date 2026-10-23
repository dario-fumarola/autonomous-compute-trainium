# Ongoing research objective

The owner authorized continuing this research across days on October23,2026,
with periodic steering reviews. Seek a substantial, independently reproducible
benefit from teacher-free latent computation during natural-text pretraining.
This is an ongoing objective, not a promise that a positive effect exists.
Negative and inconclusive results stay part of the evidence.

The current frozen variable-depth campaign will finish under its existing
protocol, source hashes, decisions and04:15 UTC cutoff. That cutoff bounds this
campaign, not the owner's longer research authorization. Later experiments
receive new identities, fixed budgets and appropriate qualification before
launch. Do not extend a failed recipe indefinitely or scale a model simply
because a positive result is desired.

## What counts as progress and success

Dependence on a second pass, better prediction than an unsupported truncation,
or a gain bought only with more parameters is insufficient for the overall
objective. Seek a useful gain against a healthy, independently tuned ordinary
baseline, with source-token and parameter controls and a separate training-
compute comparison. Include inference cost. The current minimum useful effect
is0.01 nats per real source token. Domain-specific positives may be worthwhile
intermediate findings, but must not be relabeled as a general-text win.

A promising discovery must receive an independent confirmation: freeze the
candidate, baselines, data rules, LR, endpoint, inference path, primary outcome,
seed count and analysis before opening another held-out panel. Use new source
groups and at least five fresh paired training seeds for a terminal positive
claim. Choose the seed count from development variance and measured budget,
not from repeatedly checking confirmation outcomes. Report seed variation and
source-group uncertainty distinctly. No dropping failed or unfavorable seeds,
selecting favorable exits, or changing the primary domain weights afterward.

## Repeated testing

The completed legacy studies and the currently frozen study retain their own
predeclared analyses and scope. A positive found in the current study is a
candidate for the ongoing objective, not automatic terminal success.
Development experiments may use designated TUNE/DEV repeatedly, with that
adaptation disclosed. Consumed TEST panels cannot confirm new interventions.

For future terminal confirmation attempts, allocate the family error budget
before each attempt: alpha_k =0.05/2^k for k=1,2,..., so total allocation is at
most0.05. Record every attempt and spent allocation, including failed ones.
Any multiple primary comparisons within an attempt must share its allocation.
Use fixed-sample intervals/tests at that allocated level and require the
predeclared meaningful-effect criterion, rather than stopping at the first
nominal p<0.05. Independent data and seeds are essential to this procedure;
alpha spending cannot repair leaked or outcome-selected confirmation data.

The separate `compute_trainium/terminal_confirmation.py` helper validates
future one-primary registrations and computes fixed-sample paired Student-t
intervals for five or more seeds at the allocated level. Its tests check the
quantile, freshness constraints and allocation sequence. It uses host SciPy
(currently1.18.1); it is not part of the running native trainer or this study's
frozen analysis. Seed intervals assume independent, approximately normal
paired-seed differences and are conditional on the chosen panel. Source-group
uncertainty, the data gate and independent numerical audits remain necessary.
No terminal attempt has been registered and no new TEST panel has been opened
by adding this helper. A valid numerical interval alone cannot establish the
resource and scientific controls required for a final claim.

## Operating constraints

Keep the natural-source policy: no generated research corpus, external teacher
or synthetic reasoning supervision. Numerical software fixtures are separate
from research data. RL, learned routing or further architectural mechanisms
need their own explicit scientific protocol; they are not added to a running
comparison. Reviews provide hypotheses to check, not instructions to accept
uncritically.

Use this instance effectively, with measured throughput, bounded qualification,
full-state recovery and all available chips when independent useful work is
ready. Preserve512 GiB host MemAvailable and35 GiB free on every output
filesystem. Count tuning, failed/stopped jobs, compilation, scoring and CPU-only
tails; distinguish training-update matmuls, wall time and reservation cost.
Preserve prior results and indispensable states. No upstream submission or
third-party message is authorized by the research objective alone.

Continue after informative negatives by choosing a new falsifiable mechanism
or a justified control. Do not manufacture a positive, hide a negative, weaken
the baseline or redefine the success criterion to match an observed result.

## Primary-literature check, October23

These papers inform possible later choices; none changes the live protocol.
The published Huginn model uses a four-layer recurrent core between two-layer
prelude/coda modules. Our16-layer shared core is a different constraint. A
smaller shared fraction is a possible controlled follow-up, not a claimed
reproduction or a demonstrated fix.
[Huginn paper](https://arxiv.org/html/2502.05171v2).

That paper also synchronizes sampled depths across workers to avoid waiting
for the deepest worker. Our static heterogeneous depths avoid retaining three
training graphs per rank in the native beta, at a measured wall-time cost.
Locked-step sampling is a possible later optimization, but it changes
per-update depth composition and needs new qualification; it is not a
drop-in change to this frozen comparison.

DeepLoop reports natural FineWeb-Edu pretraining for50B tokens. Its main loss
table has single training seeds, compares residual parameterizations, and
does not establish our desired resource-matched result. Its initialization
sweep also reports failed runs stuck near a unigram floor. That makes our old
collapse compatible with a training-dynamics problem; it does not identify
the cause of our failure or validate copying the new normalization blindly.
[DeepLoop paper, sections5.1 andC](https://arxiv.org/html/2607.13491).

Source-Centered State Evolution explicitly constrains recurrence around an
input-dependent anchor. Its direct-web comparisons improve on other shared
models, but the reported unshared eight-layer references still have lower
perplexity; those references are not parameter matched. The direct-web22M
settings use20.48M training tokens. These are useful mechanism hypotheses,
not proof that an anchor will beat our stronger ordinary baseline.
[SCSE paper, appendix direct-web tables](https://arxiv.org/html/2607.27656).
