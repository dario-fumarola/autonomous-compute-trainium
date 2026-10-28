# Does a second trained step use newly computed earlier states?

The previous branch mostly learned a prediction correction that survived
independent memory during training. This separate bounded mechanism tests
communication between intermediate states,not more training of that result.
It is a small tied-depth representation model on frozen pretrained features;
it is not self-generated discrete thought,autonomous allocation,from-scratch
pretraining,or evidence of reasoning by construction.

Both new arms share655360 parameters,initial seed7285,frozen C0/head,width128,
the original8192 TRAIN views/4315 groups,2048 AdamW updates,batch4,32 uniformly
sampled natural labels,query RNG7286,LR3e-4,WD0.1,and clipping1.0. Zero output
preserves exact raw prediction initially. No LR or source-mixture search.
Every256 updates contain128 one-step and128 two-step updates,shuffled with
independent Python RNG7299. The complete schedule is frozen before fitting;
both arms use the identical depth and target schedules. Each trained checkpoint
explicitly supports exits1 and2. No extra exit loss or target upweighting.

The first cell application computes causal latent states from frozen final
representations. The static arm refines queries again using original earlier
memory;the written arm computes first-step states at every preceding position
and lets the second step attend to those updated states. BOTH arms preserve
the same updated current-token slot on the second pass. Thus the intervention
is earlier-state writing,not merely changing the query's own memory slot.
All attention remains causal. Memory is local to each input window,not a
persistent cross-document memory or a claim about cached deployment.

After fitting and mature audits,score both trained exits on all448 reused TUNE
views. Retain the independently trained one-step model,local control and raw
C0 scores. On the written model's two-step exit also compute the first-step
writes but suppress earlier writes at the second read,retaining its updated
current slot. This intervention is conditional on fixed weights and may be
out of distribution;the separately trained static arm provides its learning
control. It is not sufficient on its own to establish reasoning.

Report the full model-by-exit table,all domains,and source-group conditional
intervals. Primary useful-iteration gate retains the original practical bars:
written-two must improve raw NLL by>=0.01 and beat the independently trained
one-step model by>=0.005;it must also beat its own supported shallow exit and
the new static-two control by>=0.005 each. Require positive lower95% bounds
for those paired improvements and for the earlier-write intervention effect.
All criteria must pass. A positive result is only a development lead requiring
fresh sources/training seeds and a conventional full-work comparator.

No reserve,TEST or new dependency panel is scored. The320 unscored reserved
groups stay excluded. Equal target exposure,parameters and update count do
not imply equal work:count per-depth forward/backward matrix operations,
including the vocabulary projection and all earlier-state writes. Reused C0,
feature extraction,old one/local controls and prior diagnostics remain costs.
Do not infer efficiency from CPU execution speed or omit the pretrained base.

Before fitting,check exact zero-output identity,one-step parity with the
qualified branch,causality,independence from other selected queries,all named
gradients against an explicit per-query reference,and the intervention's
agreement with static memory at identical weights. Qualify full-size FP32
against FP64 at the unchanged2% gradient/.0002 loss gates on one fixed TRAIN
view/domain at existing mature weights;those weights are disposable and are
never scientific initialization. Repeat at both mature endpoints,restore
full model/AdamW/counter/RNG,and replay one disposable update exactly.

Two16-thread CPU fits may overlap only with existing resource reserves intact.
40-minute overall registration;each fit has20minutes and must project its
remaining duration after16updates. Cache reuse requires no new Native pass.
Keep512GiB host availability,35GiB output-disk space,10000 inodes and48GiB RSS
per process. Preserve full states and failures. No score-based extension.
If this screen fails,close this frozen-feature two-step mechanism;do not add
more steps or search coefficients on the same evaluation panel. Neither
outcome automatically admits a larger backbone,router or confirmation.
