# Exact cached decode has a separate cost

The generated-state51.75M model executes two dependent six-layer body calls per
natural prediction; the51.18M ordinary model executes one eleven-layer call.
Each projects to the vocabulary once. A latent and source position have separate
KV entries even though they share a rotary index.

| Supported source context | Latent forward matrix FLOPs | Ordinary forward matrix FLOPs | Latent overhead |
|---|---:|---:|---:|
|1|108,695,552|102,356,992|6.19%|
|128|114,937,856|105,218,048|9.24%|
|256|121,229,312|108,101,632|12.14%|
|512|133,812,224|113,868,800|17.51%|

At batch1 with FP32 caches, KV storage is48KiB versus44KiB per source token:
24MiB versus22MiB at context512. That is9.09% more cache storage for the latent
model. Lower-precision cache figures would halve the bytes, but such a serving
implementation is not qualified by this inventory.

These are analytic dense-operation and raw-storage counts, checked against
executed linear/attention shapes in three numerical software tests, including
MHA and GQA. They exclude elementwise work, allocation/copy costs and kernel
overhead. They are neither latency/throughput measurements nor a prefill result.
Exact latent prefill has sequential dependencies; approximate Jacobi prefill
must not inherit the exact readout's quality without its own checks.

No accelerator, training update or held-out model score is used. This inventory
does not change training-work budgets or promote a scientific claim. The
matched-training-work ordinary control remains the outstanding comparison.

[Complete counts and source bindings](report.json).
