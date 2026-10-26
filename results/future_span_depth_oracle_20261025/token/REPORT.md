# Token-level output selection has substantial within-family headroom

The retrospective token-output oracle is complete. All six original depth scores reproduce exactly, and every original document score and target position passes the registered checks. A separate minimum-identity/domain-sum calculation reproduces both oracle losses within 1e-12.

| Seed | Best uniform depth3 | Document oracle | Token oracle | Original matched-work One | Oracle advantage over One |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5231 | 2.586512654 | 2.586340621 | 2.503926265 | 2.581289557 | 0.077363292 |
| 7283 | 2.584444974 | 2.584254559 | 2.502679342 | 2.581382256 | 0.078702914 |
| Mean | 2.585478814 | 2.585297590 | 2.503302803 | 2.581335907 | 0.078033103 |

Allowing each actual target to select its smallest loss among depths1/2/3 improves the recorded outputs by about0.082 nats beyond the document oracle. These are authentic consumed-DEV targets, with all754,750 targets,603 documents,518 source groups and fixed60/20/20 domain weights retained for each seed. The weights and model states are fixed. The oracle knows the actual target; it is not a trained or deployable predictor.

The strongest new ordinary8 reference scores2.480522151, which is0.022780652 better than the token oracle mean. Its training matrix budget is1.955151 times that of these old sampled-depth checkpoints, and its second training seed is17293 rather than7283. That is a descriptive higher-budget reference, not an equal-training-budget or paired-seed comparison. The original One controls above are the proper same-seed, approximately work-matched comparison.

A selector restricted to these recorded output distributions cannot beat the present ordinary8 scores on this finite panel even with perfect target-informed choices. This does not bound a jointly trained router, altered mixed-depth hidden trajectories, other checkpoints or architectures. Conversely, the positive0.0780 within-family upper-bound gap does not show that a prefix-only policy can predict or realize the benefit, or that sparse execution earns its costs. No adaptive-computation family is declared impossible, and no router is admitted by this result.

The difficulty strata that motivated this readout compare independently trained fixed-two and One models. They are not same-checkpoint token-level interventions, and they are separate from these older sampled-depth models.

The needed token arrays had not previously been saved for these checkpoints. Six unchanged CPU position readouts regenerated them, using two CPU workers while the admitted lean One training continued on all16 chips. Replay wall time was18.63 minutes, child CPU time6.478 core-hours, with6,300 window forwards,zero optimizer updates andzero accelerator chips assigned. Testing took40.45 seconds (324 passed,2 skipped). CPU replay overlaps native training; reservation time must not be double-added. Raw texts,tokens,weights and position arrays remain private.

This is retrospective development analysis, not a primary result, an efficiency win or confirmation. No fresh reserve,TEST or terminal alpha was used. The current4B gate remains unmet. Finish the remaining declared One controls; retain ordinary8 as the strong comparator.

[Full result and cost receipt](report.json) · [Independent verification](independent-verification.json) · [Registration](registration.json).
