# Same-weight recurrence diagnostic

This completes the fixed 15-cell inference matrix on the unchanged curve DEV
panel. Each checkpoint has the same weights and token positions across its
readouts. Distributed trained-path calibration reproduces every original score.
The original three-seed 1B negative result remains unchanged.

| Training path | Inference core passes | Mean NLL |
| --- | ---: | ---: |
| ordinary | 1 | 2.519907 |
| ordinary | 2 | 2.642202 |
| recurrent | 1 | 3.050335 |
| recurrent | 2 | 2.550448 |
| recurrent | 4 | 2.693321 |

| Intervention | NLL improvement, first path minus second | 95% paired-seed interval |
| --- | ---: | ---: |
| Recurrent checkpoint: 1 to 2 | 0.499887 | [0.479285, 0.520489] |
| Ordinary checkpoint: 1 to 2 | -0.122296 | [-0.136731, -0.107860] |
| Recurrent checkpoint: 2 to 4 | -0.142873 | [-0.147262, -0.138485] |

The recurrent model's one-pass readout is outside its training configuration.
A two-pass advantage establishes dependence on the trained computation path;
it does not establish superior training, an interpretable algorithm, or that
further iterations should improve the prediction. Four passes is a fixed
extrapolation diagnostic. Intervals are exploratory and not simultaneous.

The accompanying fixed-prefix CPU readout uses the same isolated-document
format for TRAIN and DEV, with 512 real targets from each of four source
documents/domain/split. TRAIN prefixes come from the beginning of the actual
frozen training order and are seen by all terminal models. This small sample
is a diagnosis, not a population training/generalization-gap estimate. Compare
the architectures on the same examples; do not interpret a raw TRAIN–DEV level
difference as an unbiased generalization gap.

Global gradient norms were logged before clipping approximately every ten
updates plus phase boundaries. The history therefore estimates sampled
exceedances, not exact per-update clipping frequency. The derivative readout
uses three fixed natural TRAIN prefixes at the unchanged seed23 checkpoint,
with equal-weight copies separating the two applications. Its gradient cosine
is observational and cannot establish that sharing caused the training loss.
State RMS, centered token variation and between-pass changes are reported for
all fixed prefixes. No synthetic research corpus, optimizer update, new LR
selection or TEST scoring is part of these measurements.

[Machine-readable comparisons](summary.json) · [Full state readouts](state-probes.json) ·
[Frozen new campaign](../../docs/MECHANISM_DIAGNOSIS_20261023.md).

On the three inspected TRAIN prefixes, aggregate application-gradient cosine is 0.227. Residual RMS rises from approximately 1.44 after the first core pass to 2.74 after the second, with substantial centered token variation in both. These summaries provide no evidence of gross token-state collapse; they do not establish a causal optimization explanation.

A separate fixed 32-window TRAIN attention intervention retains all tokens, positions and targets. Blocking attention across document boundaries changes ordinary NLL from 2.521323 to 2.522929, and recurrent NLL from 2.548532 to 2.551067. The recurrent deficit remains in both modes. This limited sample does not support boundary context as the sole explanation of the loss. [Boundary protocol](../../docs/DOCUMENT_BOUNDARY_DIAGNOSTIC.md).
