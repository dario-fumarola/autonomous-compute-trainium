# First four-condition workspace discovery pilot

**Result: the sampled-workspace recipe loses to ordinary continuation. Its own workspaces do not yet provide a positive held-out gain.**

All four endpoints are complete and independently audited. This is a single-seed short constant-rate continuation from the same C0; it is not converged tuning, confirmation, transferable reasoning, learned allocation, or training from random initialization.

| Predictor | Updates | Next-four NLL ↓ | First-token NLL ↓ | Raw32 NLL ↓ |
| --- | ---: | ---: | ---: | ---: |
| Common C0, no workspace | 0 | 2.406378 | 2.400690 | 2.278117 |
| Ordinary continuation | 931 | 2.387626 | 2.383042 | 2.256017 |
| Learned workspace, one own sample | 48 | 2.422095 | 2.422818 | 2.271458 |
| Trained fixed pauses | 421 | 2.548699 | 2.632520 | 2.280280 |
| Own-sampled supervised draft | 63 | 2.429410 | 2.410787 | 2.274657 |

Nominal matrix work includes padded dense body, full-square attention, generation, scorer, backward, checkpoint recomputation and vocabulary projection. Controls round up. This is not hardware-time matching. A dispatch-counted software fixture verifies the model-level recomputation factors; Native lowering and nonmatrix costs remain separate.

The ordinary-minus-workspace advantage is **−0.034469 nats**, conditional source-group 95% interval **[−0.057516, −0.005504]**. For the first target alone it is −0.039776, interval [−0.085525, +0.015177]. The raw32 comparison also favors ordinary by0.015441nats.

## Did the generated content help?

At fixed workspace weights, raw next-four NLL is2.413154; adding its own prefix-generated sample changes this to2.422095. The gain is **−0.008941**, interval [−0.018593, +0.000922]. Correctly matched notes beat same-domain/different-source notes by only0.003327, interval [−0.009057, +0.015954]. Neither establishes useful content.

Fixed-content corruption is much worse, but that alone is not causal reasoning evidence: the donor contrast is weak and the separately trained pause interface also performs poorly. The draft condition does not supply a positive workspace gain either. C0 with its own sampled note scores2.421645, essentially the same absolute score as the trained workspace2.422095.

The reader remains sensitive to privileged informative content: an actual future copy improves next-four NLL by0.744383 at the workspace weights. On the first target its diagnostic gain is0.185868. These copies reveal the answer intentionally for a positive-control diagnostic; they are NEVER training thoughts, deployed proposals or answer-aware selection. They do not establish a reasoning mechanism.

## Gates, scope and costs

All179 named gradients pass independent HF-body/FP64-loss audits. Largest named relative errors: ordinary0.002141%, workspace0.000734%, pause0.001098%, draft0.001051%. Workspace frozen rewards agree within0.00000217nats. All64 complete state shards are preserved per endpoint; no fallback. Both lost48-update workspace histories remain failed and charged.

Retained fit paths use4.68865/10.58526/6.18826/5.58112 assigned chip-hours for ordinary/workspace/pause/draft. Workspace includes its819.10-second early backup plus sequential final save. The two lost histories add6.56149 assigned chip-hours; qualification, base acquisition, data preparation, CPU audits and operational gaps retain their own receipts. Do not add overlapping reservation and assigned costs. Evaluation recovery receives its own cost receipt.

The200GiB evaluation admission stops before a worker is created. A separately registered192GiB no-optimizer/no-EMA scoring worker succeeds without changing any trained weight, data, seed, panel, donor or metric. The512GiB host/35GiB filesystem reserves and2% numerical gates remain unchanged.

Evaluation:320 source-only-selected TUNE anchors,192/64/64 web/math/code,101 source groups, one prefix-only eight-token sample per anchor with fixed seeds. No best-of, resampling or future-aware thought choice. The5000-draw intervals are conditional on one fitted checkpoint pair and reused TUNE sources; they do not measure training-seed variation. Base pretraining contamination cannot be excluded. No fresh confirmation reserve was consumed.

The constant1e-5 short pilot has one LR opportunity per condition. The ordinary control spreads its NTP loss across159 targets, whereas pause/draft/workspace emphasize the final four; a horizon-weighted ordinary challenger remains necessary for a favorable later result. No learned halting, content-sensitive transfer, supported-length scaling, cached decode or serving-cost advantage is established.

## Decision

Do not scale this sampled-workspace recipe. Record this negative unchanged. A separately registered inference-only readout diagnostic may test greedy sampling and a prefix-tail replay interface on the same frozen models. It must report all variants, retain this original table, and remain outcome-informed development rather than confirmation. No new fit follows automatically.

[Complete scores and conditional analysis](report.json) · [Frozen recovery](plan.json) · [Evaluation costs](execution-costs.json) · [Aggregation check](aggregation-check.json).
