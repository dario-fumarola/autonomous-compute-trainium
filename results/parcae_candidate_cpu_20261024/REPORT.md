# CPU whole-model reference qualification

The candidate matches pinned upstream logits and every parameter gradient
exactly in seven checked cases. Two use the actual143M/144M parameter widths
with128-position fixtures; five cover smaller ordinary/recurrent graphs and
different forward/backward depths. All causal checks pass. The PyTorch matrix
operation counter matches the independent head-inclusive ledger in every case.

Final qualification:22.07s including imports,3.47GiB peak RSS, zero accelerator
work, zero optimizer updates. Earlier attempts and their narrower measurements
are preserved. No scientific learning or compute-efficiency result is claimed.

[Exact qualified report](attempt3-qualified/report.json) ·
[Receipts](execution-costs.json) ·
[Implementation scope and remaining gates](../../docs/PARCAE_CPU_PORT_20261024.md).
