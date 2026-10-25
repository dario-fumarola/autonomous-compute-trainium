# Future-span objective: prospective mechanism study

Status: CPU implementation/preparation only. No native qualification or new
scientific training has launched. Finish the frozen small-core pilot and its
declared readouts first. This proposal changes the order of future research,
not the results or success criteria of completed experiments.

## Decision and evidence

The next useful intervention should change the learning signal rather than
add another architecture. Test whether anticipating a short natural future
increases the value of recurrent computation. A generic auxiliary-objective
gain is a separate outcome and could justify retaining only the one-pass model.

The primary literature supports investigation,with considerable limits:

* [Future Summary Prediction,eqs.9–10 and appendixD](https://arxiv.org/html/2510.14751v2)
  defines a teacher-free multi-hot presence target over positions t+2 through
  t+tau,using reweighted binary cross entropy. Its strongest method instead
  trains a separate reverse-LM teacher. The teacher-free target is a contributed
  baseline; its loss is not included in the paper's NLL tables. Its large runs
  use250B/1T tokens and a mixture including explicitly generated mathematics.
  Those results do not demonstrate our natural-only small-model NLL hypothesis.
* [ConceptLM,table4](https://arxiv.org/html/2602.08984v1) holds its architecture
  fixed in an objective ablation: the combined objective changes WikiText
  perplexity34.30→33.45 while OpenWebText remains12.64. Concept prediction
  without quantization is worse on both. This motivates testing the signal;
  it does not establish that any future objective works. We are not reproducing
  ConceptLM or introducing its learned codebook/decoder.
* [NextLat,table2](https://arxiv.org/html/2511.05963v3) reports a small downstream
  average increase58.82→59.21 with worse FineWeb-Edu perplexity10.52→10.88.
  These are different endpoints,not evidence that auxiliary predictive states
  necessarily improve ordinary text likelihood.

These are author-reported results,not local reproductions. No synthetic source,
teacher model,reverse-model training,generated reasoning or RL enters our test.

## The falsifiable comparison

Use the same qualified small-core family for all four cells. One-pass is the
true deterministic-one configuration with the recurrent family's identical
parameterization; recurrence uses the same saved per-sequence min-one Poisson
depths in both objective cells. It has supported exits1,2,3. Use full gradients
through all sampled steps in both recurrent cells,subject to new native gates.
This common gradient rule is a new recipe; old-model comparisons cannot isolate
the objective. Do not mix last-one gradients in one cell with full gradients
in the other.

| | NTP | NTP plus future-span presence |
| --- | --- | --- |
| Deterministic one pass | O | O+F |
| Sampled recurrence | R | R+F |

Let L denote the fixed60/20/20 held-out real-token NLL,and score recurrence at
the declared two-pass exit. The primary proposed interaction is

`I = (L(O+F) - L(R+F)) - (L(O) - L(R))`.

A recurrence-specific candidate must have I≥0.01 nats,an absolute R+F gain
of≥0.01 over O+F,and a gain of≥0.01 over R. Also require a useful supported
one-to-two inference-pass improvement within R+F. An interaction created only
by damaging O+F cannot pass these requirements. Lower future-head loss is not
a scientific success criterion. Uncertainty across source groups and independent
training seeds stays distinct.

For development,propose two paired seeds5231/7283 and tuning seed4127,after
checking the initialization ledger for freshness. This is a screening proposal,
not a significance claim. Any promising candidate needs a frozen confirmation
on new source groups and at least five fresh paired seeds with prospective
family alpha allocation under the ongoing charter. No confirmation is registered
or accessed here. The eventual primary contrasts must share that allocation.

## Concrete first objective

The numerical prototype fixes horizon32,stride16 and weight0.1. At a static
input position t,predict the presence of tokens in t+2,…,t+32. Exclude the
immediate next token. Require all positions t,…,t+32 to be real source positions;
BOS/EOS/padding invalidate that anchor. Keep every ordinary next-token target.
No auxiliary span crosses a document or window boundary. Do not read an extra
future window to fill an incomplete span. Report exclusions and coverage.

Use a separate FP32 linear768→32000 auxiliary head with bias,from the same
normalized causal hidden state as the ordinary head. It is not tied to the
ordinary logits and does not feed future labels or auxiliary predictions into
the language model. At1024 positions this executes63 head slots per window,
including invalid anchors. Deduplicate repeated token IDs into binary presence.

The auxiliary loss per valid anchor is the vocabulary-summed BCE divided by31.
Average over the entire optimizer batch's valid anchors,and add0.1 times that
average to global real-token-normalized NTP. Use neither vocabulary-mean BCE
nor per-microbatch means. There are no corpus-fitted IDF weights. The head has
normal(.02) weights and constant bias log(p/(1-p)),p=31/32000;this fixed
uninformative prior prevents32000 initial negative labels from dominating.
Head draws use a separate RNG stream cloned within each paired seed.

Both zero-weight NTP controls retain the same auxiliary parameter inventory.
The prototype computes their head/loss with multiplier zero;CPU tests require
exact NTP backbone gradients and zero auxiliary gradients. Any native pruning
must be counted as actually executed work rather than assumed dense work.
The head adds24,608,000 training-only parameters,166,571,840 total versus
141,963,840 deployed. It uses the explicit AdamW head group in every cell.
Inference restores the existing causal model with the auxiliary head removed.

This is an inexpensive,explicit adaptation of the hand-crafted FSP target,
not a release reproduction. Sparse anchors,a plain linear head,constant
normalization,no IDF,and our model/data/precision all differ. Failure would
close this particular proposal at the declared budget,not every predictive
latent architecture.

## Qualification and finite expenditure

The existing pilot keeps all16 chips until its registered stages finish. CPU
software checks now cover target offsets,presence rather than counts,document
boundaries,empty masks,global DDP/accumulation normalization,head removal,
zero-weight equivalence,and future changes leaving prefix predictions intact
at each supported depth. Numerical arrays are software fixtures only.

A separate eight-window TRAIN-only screen compares the candidate with a clean
pinned upstream FP32 backbone plus an independently written presence loss.
It makes zero optimizer updates and accesses no held-out scores. This cannot
substitute for complete-global native gradients,first updates,full-state/RNG
recovery,replica equality and100 varying-schedule warm updates for the new
head/loss/full-gradient paths. Preserve the existing2% gradient/0.003-nat loss
gates and resource reserves. Stop on nonfinite gradients;never silently skip
overflowing scientific updates.

Before any new fit,register the measured budget,source hashes,full execution
queue,initial-state hashes,LR opportunities and endpoints. The proposed main
exposure is1,004,887,590 real targets per cell,with each own completed50%
cooldown. Give every cell the same finite three-rate selection opportunity
(.004,.008,.016) at301M targets. Boundary winners remain unbracketed;no added
rates,horizons,weights or depths after seeing outcomes. Feasibility measurement
may reduce the common exposure before any new scientific scores,not afterward.

Count all forward/backward work,attention,the normal and auxiliary projections,
all sampled recurrent steps,all replicas' optimizer matrix operations,padding,
branches,tuning,qualification,failed attempts and CPU scoring. Report assigned
chip-hours and whole-instance opportunity cost separately. The first comparison
is equal-data mechanism screening,not a claim of matched-training efficiency.
A promising result earns a separately annealed compute-matched comparison with
the strongest eligible one-pass model,including O+F and the cheap connection.

Carry forward the syntax-selected natural-code context diagnostic as exploratory
readout. Include the separately trained one-pass control prospectively. Require
absolute full-context benefit as well as a full-versus-short-context interaction;
an interaction caused only by damaging short-context performance is insufficient.
Consumed panels cannot become fresh confirmation by changing the objective.

If the future objective helps both architectures equally,retain that learning
result without claiming useful recurrence. If only the auxiliary head improves,
or the bounded comparison fails the absolute/interaction conditions,close this
objective proposal. Do not automatically start another horizon/weight/depth
search or scale up. Preserve the existing negative efficiency evidence.
