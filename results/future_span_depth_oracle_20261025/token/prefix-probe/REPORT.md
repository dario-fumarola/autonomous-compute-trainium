# Past-surprisal probe does not recover a useful selection gain

One fixed, retrospective predictability probe fitted separate low-capacity regressions in five source-group folds for each old sampled-depth model. Current/future target losses, other-depth loss features and document/source/domain IDs were excluded from predictors. Only prior depth-one surprisals within the same window, prefix position and a history-present flag were used.

| Base training seed | Constant depth3 NLL | Cross-fitted selection NLL | Improvement |
| --- | ---: | ---: | ---: |
| 5231 | 2.586512654 | 2.586511313 | 0.000001341 |
| 7283 | 2.584444974 | 2.584444974 | 0.000000000 |
| Mean | 2.585478814 | 2.585478143 | 0.000000670 |

The training-fold-chosen constant also selects depth3 throughout. The fitted policy chooses depth3 for all754,750 targets in seed7283; in seed5231 it chooses depth2 for1,050 targets and depth3 for the remaining753,700. It never selects depth1. The point-gain-above0.01 rule fails in both seeds. This closes the declared cheap past-surprisal feature family;no feature or penalty sweep follows automatically.

The token oracle still scores2.503302803 on average. This gap is an upper bound with target knowledge. Simple fixed mixing and this single causal feature family do not realize it. The probe does not test all prefix representations, current hidden-state features, jointly trained routing or changed mixed-depth trajectories. No native routing campaign is admitted.

All five source folds are fixed by source-group hash, keeping every occurrence of a group together across domains. Feature normalization and regression use training-fold rows only; held-out labels are used only for evaluation. The intercept is unpenalized and the fixed ridge coefficient is0.001. Histories reset at document and1024-position window boundaries. Five focused tests verify causal lags,window/document resets,group semantics and held-out-label exclusion;the full suite passes329 tests with2 skips.

This is analysis on already-consumed DEV,not fresh confirmation or a new language-model training result. Ten small statistical fits used2.122 wall seconds,7.694 CPU seconds and0.779GiB peak RSS,with zero language-model forwards,language-model updates or accelerator chips. Prior depth-one probabilities would have inference costs in any future implementation. The selected outputs come from recorded uniform-depth forwards;deployable sparse execution and its costs remain unestablished. Parameters are not exported as a deployed policy.

[Explicit protocol](registration.json) · [Full result and costs](report.json) · [Causality and leakage tests](../../../../../tests/test_prefix_value_probe.py).
