# Retained model paths and resource controls

Per-model retained training paths only, including launch/CPU-probe/checkpoint wall overhead but excluding separate DEV/TEST scoring. The ordinary1B decay branch is not an ancestor of its1.7B control. Shared prefixes appear in both model rows; DO NOT sum those rows as campaign cost. The phase union deduplicates shared ancestors but still excludes unused/stopped attempts, tuning, readouts and CPU-only tails, all recorded in the separate execution ledger. Update matmuls and known native-probe work are separate; elementwise/optimizer/communication/compiler/CPU math is not in those FLOP counters. Inference figures are estimated executed forward matmuls, not generation latency.

| Arm | Seed | Endpoint | Real targets | Estimated update matmuls | Retained launch minutes | DEV NLL |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| ordinary | 103 | 1B | 1,004,887,590 | 1.09841e+18 | 27.37 | 2.599135 |
| ordinary | 103 | matched | 1,699,926,288 | 1.85815e+18 | 42.74 | 2.495278 |
| fixed | 103 | 1B | 1,004,887,590 | 1.85839e+18 | 38.50 | 2.548269 |
| mixture | 103 | 1B | 1,004,887,590 | 1.85839e+18 | 50.20 | 2.587427 |
| ordinary | 211 | 1B | 1,004,887,590 | 1.09841e+18 | 26.70 | 2.598596 |
| ordinary | 211 | matched | 1,699,926,288 | 1.85815e+18 | 42.29 | 2.495164 |
| fixed | 211 | 1B | 1,004,887,590 | 1.85839e+18 | 38.87 | 2.545188 |
| mixture | 211 | 1B | 1,004,887,590 | 1.85839e+18 | 50.14 | 2.590665 |
| ordinary | 307 | 1B | 1,004,887,590 | 1.09841e+18 | 26.71 | 2.601686 |
| ordinary | 307 | matched | 1,699,926,288 | 1.85815e+18 | 42.32 | 2.496952 |
| mixture | 307 | 1B | 1,004,887,590 | 1.85839e+18 | 49.66 | 2.587213 |

11 completed training paths; 1 still pending.

[Machine-readable paths](summary.json) · [Whole-campaign execution ledger](../execution-costs.json).
