# Frozen-predictor residual workspace screen

**The registered iteration gate fails.** A residual branch can learn improvements on its TRAIN labels while preserving the ordinary predictor exactly. Those improvements barely transfer to reused held-out sources, and two steps do not beat the cheaper linear correction. No useful-iteration or efficiency claim follows.

Unlike the closed whole-backbone feedback recipe, only this new small branch learns. C0 and its vocabulary head stay frozen. The branch operates in final-representation space with exact zero-output initialization. One and two cell executions use the same parameters and initial values; the linear control is a cheaper low-rank correction of normalized features.

| Condition | Branch parameters | Held-out NLL | Held-out gain over frozen C0 | Conditional 95% interval | Gain on trained labels |
| --- | ---: | ---: | ---: | --- | ---: |
| Frozen C0 | 0 | 2.282112205 | — | — | — |
| linear | 524,288 | 2.280833081 | +0.001279125 | [-0.001774290, +0.003945696] | +0.056549 |
| one | 655,360 | 2.281997770 | +0.000114435 | [-0.003474401, +0.003016605] | +0.054943 |
| two | 655,360 | 2.281367359 | +0.000744846 | [-0.002580009, +0.003540164] | +0.041834 |

The two-step gain is 0.000745 nats, against a preregistered 0.01 development threshold. Its advantage over one step is 0.000630, interval [−0.000171,+0.001406]; against linear it loses 0.000534, interval [−0.002219,+0.001104]. Both also miss the required 0.005 gain with positive lower interval. All intervals resample source groups conditional on these fitted models, not training seeds.

The TRAIN diagnostic is explanatory and was registered after the fixed screen finished. It uses the first 512 training views and does not select a checkpoint or hyperparameter. Gains on those actual trained labels are about 0.04–0.06 nats. On the same TRAIN views at the final 32 positions, gains are 0.01339/0.01276/0.01077 for linear/one/two; these positions are not all directly supervised in the sparse sample. The smaller matched-position TRAIN gains and the near-zero held-out gains support a generalization problem in this screen. They do not distinguish sample size, regularization, optimization and capacity, or prove an intrinsic limit on iterative computation.

Each arm processes 2,048 authentic TRAIN views from 1,544 source groups, with 1,229 web/410 math/409 code views, and no group overlap with the 448 evaluation views. There are 512 updates, four views per update, and 32 uniformly sampled targets per view: 65,536 natural label exposures per arm. The evaluation uses the existing 448-view, 116-group panel and fixed 60/20/20 domain weights. It is reused development evidence, not confirmation. LR3e-4, AdamW weight decay0.1, clipping1.0 and seed7285 were fixed before fitting. No tuning, intermediate quality selection or endpoint extension occurred.

Thirteen focused CPU tests pass. Native frozen-feature extraction on all16 chips passes three independent full-size HF references, with maximum feature error0.000103 and logit error0.0000477, unchanged parameters and no operator fallback. Full FP32 features are retained. The common raw32 score differs from the earlier C0 result by9.44e-7, below the frozen1e-5 tolerance. Each trained branch restores exact parameter, optimizer-moment and sampling-RNG values. Every named gradient at an authentic full-size training input agrees with CPU FP64; maximum relative error4.734e-6, below the unchanged2% limit. These checks establish implementation consistency, not the mechanism claim.

| Arm | Recorded warm seconds/update | Measured matrix operations/update, including frozen-head backward | CPU fit plus evaluation wall seconds |
| --- | ---: | ---: | ---: |
| linear | 0.178152 | 106,094,919,680 | 117.382 |
| one | 0.182680 | 106,326,720,512 | 119.890 |
| two | 0.179823 | 106,558,521,344 | 118.524 |

The comparison is equal data, not matched total training work. The vocabulary head dominates these small-branch updates; adding the second cell changes measured matrix work by about0.22%. Vector operations, softmax, data movement and wall-clock utilization are separate from this matrix ledger. The all-chip feature phase takes50.036seconds including launch/closure, costing0.222383 assigned chip-hours. The three CPU fits run concurrently; their process wall times are not additive elapsed time. The later CPU TRAIN/recovery/gradient diagnosis takes127.433seconds and changes no saved checkpoint. The common pretrained foundation and30-step reader bootstrap remain shared prior costs, not free or reproduced training. No inference latency or end-to-end efficiency advantage is established.

This screen gives a reusable way to isolate branch learning without degrading the base predictor. It produces no winning iterative model. A further data-volume or task-structure experiment would require its own bounded design and keep the linear/single-step controls; it cannot be presented as continuation of a successful gate. No full-model fit, router, confirmation or from-scratch campaign is admitted.

[Protocol](../../docs/FROZEN_RESIDUAL_SCREEN_20261028.md) · [Frozen plan](plan.json) · [All results](report.json) · [TRAIN and numerical diagnosis](train-diagnostic.json) · [Execution costs](execution-costs.json)
