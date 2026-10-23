# Untied replication and annealed compute control

Two fresh seeds use the previously selected recipe, without further tuning.
Seed 23 is the discovery model and is retained. Every endpoint completes its
declared decay. The untied model has 251.1M parameters; ordinary/shared have
150.4M. No synthetic research examples, teacher or RL enter this comparison.

| Model | Seed | Endpoint | Real targets | DEV NLL |
| --- | ---: | --- | ---: | ---: |
| ordinary | 23 | 1B | 1,004,887,590 | 2.518071 |
| shared | 23 | 1B | 1,004,887,590 | 2.549316 |
| untied | 23 | 1B | 1,004,887,590 | 2.489247 |
| ordinary | 47 | 1B | 1,004,887,590 | 2.517956 |
| shared | 47 | 1B | 1,004,887,590 | 2.547870 |
| untied | 47 | 1B | 1,004,887,590 | 2.494753 |
| ordinary | 71 | 1B | 1,004,887,590 | 2.523693 |
| shared | 71 | 1B | 1,004,887,590 | 2.554158 |
| untied | 71 | 1B | 1,004,887,590 | 2.487502 |
| ordinary | 23 | 3B | 3,014,637,826 | 2.336914 |
| shared | 23 | 3B | 3,014,637,826 | 2.336800 |
| ordinary | 23 | matched FLOPs | 1,699,926,288 | 2.421944 |

Positive gain favors the second model.

| Paired comparison | Mean gain | 95% training-seed interval |
| --- | ---: | ---: |
| ordinary_to_shared_three_seeds | -0.030541 | [-0.032202, -0.028881] |
| ordinary_to_untied_three_seeds | 0.029406 | [0.013225, 0.045586] |
| ordinary_to_untied_fresh_seeds | 0.029697 | [-0.052818, 0.112211] |
| shared_to_untied_three_seeds | 0.059947 | [0.043129, 0.076765] |
| shared_to_untied_fresh_seeds | 0.059886 | [-0.026127, 0.145899] |

| Single pair | Gain | 95% source-group interval conditional on weights |
| --- | ---: | ---: |
| ordinary_to_shared_3b | 0.000114 | [-0.002381, 0.002077] |
| ordinary_matched_flops_to_untied_1b | -0.067303 | [-0.072733, -0.062260] |

The original ordinary/shared three-seed outcome remains primary. The combined
untied interval includes its discovery seed; the separate fresh-seed interval
has only two fits and one degree of freedom. A token-matched gain for the larger
untied network is not a parameter- or compute-matched win. The ordinary resource
control matches estimated cumulative dense training matmul work, including the
vocabulary projection, with integer-update rounding. It does not equalize Adam,
communication, compilation or measured chip-hours. The resource comparison is
one seed; its group bootstrap cannot establish robustness to training seeds.
All domain scores and exact work counts are retained in the machine-readable
readout. TEST is a separate frozen confirmation, never a selection panel.

[Protocol](../../docs/CONFIRMATION_FOLLOWUP_20261023.md) · [Full readout](summary.json).
