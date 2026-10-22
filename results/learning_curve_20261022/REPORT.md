# Natural-text learning curve — running

LR selection is complete and the main comparison is running. The first paired seed has a 100M held-out point; it is preliminary, with the other seeds and longer milestones still pending. The [protocol](../../docs/LEARNING_CURVE_PROTOCOL.md), [specification](specification.json), [corpus manifest](corpus-manifest.json) and [frozen choices](selected-lrs.json) were fixed before curve evaluation. TEST is sealed.

Compare ordinary and early-pause 150,436,608-parameter random models, seeds 23/47/71, on identical real targets and one frozen data order. The milestones are 100,488,104 / 301,465,908 / 1,004,887,590 real targets per run. Only the final endpoint is the primary test; both arms finish their decay. Intermediate stable-trunk curves are descriptive.

## Preliminary 100M point

Seed 23: ordinary NLL **3.704642**, early-pause NLL **4.013455**; ordinary minus pause **−0.308813 nats**. This one early pair favors ordinary. No paired-seed interval is available yet. Both score the same 614 isolated documents / 777,348 real targets. The final 1B endpoint across all three seeds remains primary.

![Learning curve](learning-curve.png)

[Aggregate records](summary.json), [domain curves](domain-curves.svg), and [source-group uncertainty](source-group-uncertainty.json). Source-group intervals are conditional on the fitted models and do not substitute for training-seed uncertainty.

## Independent tuning

Each candidate used seed 11 and the same first 100,488,104 targets, with its own completed short decay. Only a separate tuning cohort was scored. Lower is better.

| Peak LR | Ordinary tune NLL | Pause tune NLL |
| --- | ---: | ---: |
| 0.0003 | 4.058754 | **4.127852** |
| 0.001 | **4.051340** | 4.160350 |
| 0.003 | 4.848056 | 4.926001 |

These are tuning scores, not the main scientific comparison. All six trials passed CPU numerical checks, full replica-state equality and clean supervised exits. The [protocol audit](protocol-audit.json) checks hashes, paired initial weights, exposure and selection. Main weights never reuse tuning weights.

## Compute and scope

The 16-chip qualification measured 855,336 real targets/sec ordinary and 695,948 with pauses padded causally to 1536. Padding follows every scored position and passed CPU output/full-gradient and native checks. Count all padded work: the pause arm costs 1.562× the ordinary dense training matmul estimate, including the head. This experiment is token-matched; it cannot establish FLOP efficiency or isolate thought content.

The [budget](launch-manifest.json) estimates 2.40 steady node-hours including tuning, with a 4.8–7.2 h planning allowance and an eight-hour campaign cap. The [ledger](resource-ledger.json) counts actual training, DEV, qualification and failures separately from the inclusive node window. Source preparation took 296.98 s and overlapped qualification on this same instance.

The corpus uses pinned natural web/math/code in a 60/20/20 mixture. Split by source group, exact-deduplicate and apply a disclosed approximate lexical duplicate screen. Held-out NLL scores every real target in isolated documents; training uses packed documents and permits causal cross-document attention within a window. No generated research examples, teacher or RL. The [token-ID audit](source-token-id-audit.json) records 44 natural code source tokens using ID 0, also used as the pause input; these remain scored real targets in both arms. Artificial boundary loss masks use provenance, not token-ID equality.

The useful-gain threshold is 0.01 nats, using a paired-three-seed 95% interval conditional on this corpus/order. At 1B targets this is about 6.7 tokens per parameter, so a negative result would describe this early exposure, not convergence. Source-group uncertainty is reported separately. No timing, autonomous-placement or textual-reasoning claim follows.
