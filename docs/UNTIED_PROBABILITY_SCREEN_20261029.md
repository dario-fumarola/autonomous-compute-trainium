# Does untying the predictive readout help probability-valued work?

A prospective screen,admitted only after complete actual-shape qualification in
[the interface protocol](UNTIED_PROBABILITY_INTERFACE_20261029.md). This is a
new structural comparison,not an extension of the failed budget-cue fit.

Three new random-initialization fits all have79,184,384parameters,including an
independent output table copied from the input table at initialization. All
share seed9511,source order,60/20/20 natural mixture,context512,global256windows,
AdamW(.9,.95),eps1e-8,clip1 and peakLR1e-3. Both tables have zero decay;body
matrices have0.1. There are no cues or auxiliary losses. Qualification weights
never initialize research training. Only original natural-token NTP is used.

| Arm | Supported extra steps | Updates | Natural targets | Matrix training work |
| --- | ---: | ---: | ---: | ---: |
| Fixed one probability replay | 1 | 1530 | 200,190,896 | 190,988,262,121,144,320 |
| Ordinary,equal data | 0 | 1530 | 200,190,896 | 85,637,181,116,252,160 |
| Ordinary,equal work | 0 | 3413 | 446,570,790 | 191,032,483,104,423,936 |

The ordinary-data work is recomputed in registration;the executed ledger is
authoritative. Vocabulary projections and probability-to-embedding matrix
products count even with shared storage. Untying adds parameters/storage,not
new matrix operations. Elementwise/optimizer/communication costs remain outside
this historical matrix ledger;measure warm and total assigned chip-time too.
At inference fixed-one uses2.2302times ordinary forward matrix work;cached
serving cost remains a separate unmeasured quantity.

Every endpoint has32-update warmup and its own final10%cosine cooldown to0.1peak.
Recovery-only midpoint snapshots at768(or1792 for work control)are not quality
readouts or new selection endpoints. Bounded four-rank capture remains qualified.
The control must be freshly trained with the same independent output table;
old tied scores cannot substitute for it.

Freeze new initial-state equality,source/corpus/model hashes and all historical
score bytes before fitting. Reuse the tied fixed-one/equal-data/equal-work
readouts as explicitly historical contrasts. Report the paired difference in
gaps:(untied ordinary minus untied replay)minus(tied ordinary minus tied replay).
An interaction without an absolute ordinary-control win is not success. Untying
adds parameters to both new arms;cross-generation contrasts are not fixed-N
estimates. The reused controls provide no extra training-seed evidence.

Primary mechanism screen:fixed-one improves over new equal-data ordinary by
at least0.01nats with positive conditional95%source-group lower bound. Efficiency
screen:the same criterion against new equal-work ordinary. Use the frozen384
broad source groups,unchanged60/20/20 weighting,2000bootstrap draws/seed9514,and
report every domain and all253 natural dependency cases regardless of sign.
No newly favorable slice or unsupported shallow exit may replace these tests.
The new ordinary conditions have the same single common LR:preliminary screening,
not an independently tuned conventional frontier. The320protected groups stay
sealed. A successful screen would still need stronger controls and replication.

Full-global mature gradients,input/output/body families,exact CPU restoration,
replica equality and independent FP64 readout gates must pass before scoring.
Same2%gradient-family and2e-4readout thresholds. The semantic permutation is an
OOD intervention and cannot establish content-sensitive reasoning by itself.
No supported inference-depth curve or learned allocation is claimed.

Bound:three fits plus evaluation,three-hour total controller deadline,one-hour
Native-phase deadline and30-minute CPU-phase bound. Reserve7GiB for full
midpoint/final states,gradients and artifacts in addition to35GiB disk;retain
512GiB host,192GiB worker and projected-inode safeguards. All failures and
qualification costs remain reported. No larger model,longer run,router or
confirmation follows automatically;another negative closes this untying screen.
