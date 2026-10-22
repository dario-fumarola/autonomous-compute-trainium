# Protocol v1 — natural text, before scientific training

The private October 22 meeting motivates learning computation during pretraining, without a larger model supplying answers or thoughts. The transcript stays private. The current owner direction excludes synthetic data.

## Falsifiable claim

At the same real source-token exposure, a model with fixed computation positions from random initialization reduces held-out real-token NLL compared with an otherwise identical ordinary model. Support a useful gain at a tested exposure only when the lower paired-seed interval bound exceeds 0.01 nats/token. An upper bound below that threshold rules out a gain of that size at that exposure; an overlapping interval is inconclusive. This threshold is declared before scientific results. A short failed learning diagnostic says nothing about eventual convergence.

The first claim concerns prediction of natural text. The separate efficiency claim requires improvement at the same cumulative estimated training FLOPs, including the LM head, attention, pauses, backward pass, and any recomputation. Report measured chip-hours as well. Never infer FLOP efficiency from a token-matched win.

## Design

Start at approximately 150M parameters with a pinned 32k tokenizer, tied input/output weights, RoPE, RMSNorm, grouped-query attention and SwiGLU. Count actual parameters. Approximately 400M and 600M are gated extensions, not simultaneous initial obligations. All models start with random weights; a tokenizer is the only pretrained asset.

| Arm | Before the source-token midpoint | After the midpoint |
| --- | --- | --- |
| Ordinary | Standard causal prediction | Standard causal prediction |
| Early pause | 16 fixed positions after each 128 source positions | Same |
| Late pause | Standard causal prediction | Introduce the same positions |

Placement follows fixed input-window positions. A document's initial BOS occupies one input position. BOS, padding and EOS contribute no real-target loss or exposure. This convention keeps shapes and target alignment fixed across arms.

Training packs documents into fixed windows with a BOS and EOS around each document. Attention is causal and may cross document boundaries within the current window; it resets at window boundaries. One input token overlaps adjacent windows, and each real target appears once, including short documents and tails. DEV uses isolated documents, identically across arms. Report this context difference. A deterministic token-deficit document scheduler targets 60/20/20; log actual per-domain token counts and freeze an adequately provisioned mixture before a scientific run. Exhausting a domain in the small qualification sample can change its final mixture. Do not claim that sample has exact mixture weights.

Within a seed, clone initial weights and retain identical real targets, window boundaries and batch order. Reserve an existing tokenizer special token for pauses in every arm, without adding parameters. Pause positions have no language-model targets. Predict the next real token from the last pause state. Late switching retains every optimizer moment and scheduler step. Learning-rate sweeps use separate checkpoints with the same starting state and data per candidate.

The early/late comparison has unequal pause exposure. It cannot isolate timing. If the pause mechanism passes, add a predeclared matched-pause-exposure follow-up before making a timing claim. A same-length control can mask every real-token key from every pause query; validate that pause hidden states are invariant to all source token changes. Deleting pauses at inference is an intervention on both computation and positions and cannot establish a content effect.

Repeated blocks are the next compute control, after the pause comparison is correct and learnable. Tune their normalization and learning rate separately. Do not equate parameter matching with compute matching.

## Data and metrics

Use a fixed 60/20/20 token mixture of natural educational web, mathematical web, and source code. Use immutable source revisions. Split by web host / code repository before tokenization and windowing. Deduplicate normalized exact content across domains and splits; audit near duplicates before a scientific launch. Check that repeated URLs and repositories cannot cross splits. Report source selection and license metadata, token counts, hashes and truncation rates. Do not publish source contents in Git.

Primary outcome: sum of real-token negative log probabilities divided by real target count, by domain, and a fixed 60/20/20 aggregate. Exclude pauses, BOS, padding and artificial boundary tokens. Also report suffix NLL versus distance from a pause, and inference latency at fixed real-token context. Keep held-out targets identical across arms. Bootstrap source groups for within-seed uncertainty; compare at least three paired seeds at the smallest size for training variance. Document intervals are not substitutes for seed variance. DEV selects learning rates; TEST stays sealed until the design and exposure are frozen.

## Qualification and exposure gates

1. CPU tests: target coverage across windows, causal future masking, pause gradients, source-group isolation, exact masked NLL and paired statistics. Test fixtures do not enter a research corpus.
2. Native hardware: CPU agreement, actual forward/backward/AdamW updates, full model and optimizer recovery, fallback inventory, fixed-shape compilation. Bound a failing native setup to 30 minutes before reassessing the preserved XLA fallback. No days of infrastructure polishing.
3. Scientific trainer gate, BEFORE LR sweeps: packed target coverage; global real-token normalization across variable masks and accumulated microbatches; DDP no_sync covering forward and backward; FP32 master parameters and optimizer states under qualified BF16 autocast; finite-gradient clipping; token-based WSD schedule; atomic full checkpoints with data/config fingerprints, cursor, exposure, optimizer, RNG and FLOP counters; fresh-process periodic DEV evaluation; exact continuation and full-topology late switch. A phase failure must stop its controller. Numerical unit fixtures are allowed and never enter the research corpus.
4. Bounded learning diagnostic on natural training documents. Track fresh DEV NLL separately from training loss. Compare all three arms before expanding. A random-model loss near log(vocab) is only a sanity check.
5. Measure at least 100 warm optimizer updates at the actual batch, precision and context, with clipping, data transfer and real-target packing included. Count repeated qualification data explicitly as throughput diagnostics, never new unique scientific exposure. Record cold compilation, evaluation and checkpoint costs separately. Use all 16 chips for a validated topology; declare token and FLOP budgets from these measurements. A 150M / 30B real-token continuation is a candidate late-training control (about 200 tokens per parameter), not an authorized automatic launch or a guarantee of convergence.

Choose learning rates separately per variant using the same candidate list and bounded tuning resources. Initial candidates: 1e-4, 3e-4, 1e-3. Use three paired seeds for the first meaningful smallest-size comparison. Keep a continuation-compatible checkpoint format and choose a token schedule before starting the comparison. Compare both fixed token milestones and FLOP milestones, with the latter selected from the measured cost model without looking at TEST. Charge tuning, evaluation, generation, preparation and stopped attempts separately.

Use a warmup–stable–decay schedule. Save the stable trunk before each intended endpoint and fork an independent short cosine-decay continuation (initially 10% of that endpoint's total training budget, final LR 10% of peak). Every compared endpoint must complete its decay. This includes the earlier token endpoints required for pause arms at equal FLOPs. The branch's tokens/FLOPs count toward its endpoint budget; additional branches count as research cost. Never compare an annealed ordinary endpoint to an unannealed pause checkpoint. Resume further trunk training from its untouched stable checkpoint, not from the annealed branch. Keep the predeclared late-switch source-token milestone fixed across branches. A FLOP endpoint before that switch receives no pause exposure and is labeled accordingly. Round endpoint and switch boundaries to shared full global updates before launch and report realized tokens/FLOPs.

The late switch is a process boundary: save the ordinary full checkpoint, exit all workers, then restore into fresh workers with the pause graph. Verify complete recovered model/optimizer state and validate the new graph's loss and full gradient against CPU before any update. DEV also compiles in a separate process. Do not retain two large compiled training graphs. The host supervisor fails closed on compiler-error logs because the current beta can return finite wrong outputs after a compiler failure. Monitor memory/disk reserves continuously and retain failures and their costs. A numerical probe and clean supervisor exit are both required; neither alone authorizes a run.

Stop scaling this mechanism if a powered smallest-size comparison and its longer continuation put the upper bound below the declared useful gain. Do not spend two additional sizes proving an unlearned task stays unlearned. If intervals overlap the threshold, report the result as inconclusive and assess the cost of reducing uncertainty. Do not claim a universal negative from one exposure.

## Later thought-content experiment

Only after compute controls and a predictive checkpoint exist, consider offline rounds: sample bounded intermediate thoughts from prefixes, score reductions in next-k-token NLL on TRAIN, filter, and update. Training on the generated thoughts is outside the current no-synthetic-corpus scope. No external teacher or REINFORCE. Thought proposals at evaluation must depend only on the prefix; no selection using held-out future tokens. Future-conditioned training selection is a training signal, not evidence of autonomous inference. Match the thought arm to a pause arm on total generation, scoring, training and inference compute. Include shuffled/content-free controls. Adaptive placement and internalization are later hypotheses.

The old 1.7B CoT-SFT extension stays a separate, frozen-result follow-up. The current no-synthetic direction excludes launching new teacher-trace supervision. No social/BDI claim, paper rewrite, 7B run or new instance purchase is part of this qualification.
