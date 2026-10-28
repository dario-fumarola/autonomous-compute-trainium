# Batch4 qualification completed

The original warm run completed104 disposable updates,including100 timed warm
updates,with all three cold numerical gates passing. The controller then stopped
before replay admission because full host allowance was unavailable. No replay
update had launched. Recovery uses the unchanged warm states,executes exactly
one replayed update,and passes exact full model/AdamW/RNG equality plus independent
whole-global CPU gradient/loss checks. All original costs remain inbatch4-v1.

Warm mean0.540454seconds for256windows/update;zero/one/two replay means
0.464921/0.542280/0.616451seconds. These are qualification timings,not science.
Qualification states are ineligible as scientific initializations.
