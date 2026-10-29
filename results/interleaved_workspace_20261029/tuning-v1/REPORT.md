# Persistent latent-state learning-rate screen

All nine fits, complete cooldowns, mature independent gradient/state audits,
and the common Native/FP64 readout pass. At128,025,288 authentic training targets
and51,747,328 parameters, generated latent states give a substantial **equal-data
development lead**. This is one seed on reused DEV, not confirmation or a
training-efficiency claim.

| Family | LR0.0003 NLL | LR0.001 NLL | LR0.003 NLL |
|---|---:|---:|---:|
| Ordinary | 3.757402 | **3.377111** | 3.467074 |
| Generated latent, fixedK2 | 3.672960 | **3.281309** | 3.615210 |
| Trained constant pause | 3.934815 | **3.391128** | 3.738495 |

Each family selects the interior LR0.001 under the frozen rule. Initialization,
authentic targets, source order, optimizer family and complete schedule are
paired. The aggregate uses60/20/20 web/math/code weighting.

| Selected family | Web | Math | Code | Weighted NLL |
|---|---:|---:|---:|---:|
| Ordinary | 3.869100 | 3.049119 | 2.229139 | 3.377111 |
| Generated latent | 3.778768 | 2.945996 | 2.124244 | 3.281309 |
| Constant pause | 3.895149 | 3.053362 | 2.216832 | 3.391128 |

The generated model improves0.095803nats over ordinary and0.109820 over the
trained pause control. Every domain favors generated states. This is stronger
predictive evidence than the preceding closed feedback/readout mechanisms.
It does not establish a reusable reasoning procedure, autonomous allocation,
or increasing utility from more supported thought tokens.

Two material limits remain. First, generated training costs4.9370times the
ordinary nominal matrix work at equal data. The selected Native retained paths
take648.22 versus396.78seconds on16 chips, a descriptive1.634times runtime
ratio. This screen has no equally work-budgeted longer ordinary fit or tuned
ordinary shape frontier. Tuning, compilation, audits and earlier implementation
failures remain charged in their own ledgers.

Second, the selected fixedK2 checkpoint fails the separately frozen TRAIN-only
sequential-consistency guard. Its learned parallel approximation does not
closely match sequential generation. The highest LR passes that guard but has
much worse prediction quality; it is not substituted as the selected winner.
No cheap sequential inference claim or larger fixedK2 fit is admitted.
[Consistency result](../convergence-v1/REPORT.md).

The common readout covers384 existing DEV source windows. Its maximum
independent FP64 per-target discrepancy is3.9324e-5nats, below the frozen0.0002
tolerance. All nine mature audits pass; both fresh evaluation reserves remain
unscored. A separately frozen exact-sequential DEV readout will measure how
much of the lead survives the intended history. A separate mixedK2/K3
qualification precedes any follow-up fit. Numerical round count is a training
approximation setting, not the number of thought tokens at inference.

[Complete aggregate receipt](report.json) · [LR selection](analysis.json) ·
[Execution ledger](execution-costs.json) ·
[Frozen protocol](../../../docs/INTERLEAVED_LR_TUNING_20261029.md).
