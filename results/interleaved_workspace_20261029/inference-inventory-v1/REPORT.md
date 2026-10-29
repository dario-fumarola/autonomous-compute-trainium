# Analytic cached-inference inventory

For512 authentic source positions at the qualified51.75M shape:

| Path | Stored positions | Vocabulary predictions | Nominal forward matrix work | FP32 KV bytes/source token |
|---|---:|---:|---:|---:|
| Ordinary |512|512|37.819 billion|24,576|
| Trained constant pause |1024|512|62.082 billion|49,152|
| Generated continuous state |1024|512|62.082 billion|49,152|

All have206,989,312 bytes of FP32 parameters. Generated states cost1.64156times
ordinary forward matrix work in this exact cached arithmetic model and twice
the KV storage. Their counted work equals the constant-pause path, but generated
states require sequential feedback; scheduling and latency need not match.

The formula charges every body projection at every stored position, one
vocabulary projection per authentic prediction, and attention over the causal
cache length at each position. ForN stored positions, attention costs
4*layers*width*N*(N+1)/2 with FMA=2. An explicit position-by-position sum matches
that formula. This differs from the conservative full-square parallel-training
ledger because the dispatched operations differ.

These are analytic counts, not measured accelerator performance. They exclude
elementwise work, memory movement, cache concatenation, allocation and launches.
The CPU exact-sequential reference is not an optimized serving implementation.
No tokens/second, end-to-end latency or hardware-time advantage is inferred.
[Complete inventory](report.json).
