# Source-separated global readout diagnostic

**The fixed latent reader is not promoted.** Calibration on unused source groups selects alpha0 for the latent model:it disables the workspace. The original fixed.5-reader negative remains unchanged. No backbone optimizer updates occurred.

Calibration uses192 authentic TRAIN-reserve views from178 groups disjoint from every60160 backbone training view and the mature-audit group. Domain counts are144 web/8 math/40 code,weighted60/20/20. Only203 eligible views existed;the initial640-view request failed before any Native run. The first192 source-order eligible views were frozen before scoring. Eight math calibration views and one reused TUNE panel materially limit inference. Upstream foundation exposure cannot be excluded.

| Model | Frozen alpha | Temperature | Held-out full32 NLL |
| --- | ---: | ---: | ---: |
| c0 | 0 | 1.0 | 2.282111699 |
| ordinary | 0 | 1.0 | 2.262856633 |
| latent | 0 | 1.0 | 2.265101872 |
| constant | 0.1 | 1.0 | 2.265023567 |

With latent computation disabled,the latent-trained raw model still loses0.002245238nats to ordinary. Conditional source-group interval for ordinary-minus-latent gain is[-0.006849120,+0.001878583];this omits seed and calibration-selection variation. Zero workspace contribution makes within-model workspace utility exactly0 by construction,not a learned adaptive-compute result.

Constant-query alpha.1 improves its own raw full32 score by only0.000109nats and does not beat ordinary. This is a tiny development readout effect,not useful learned content or a practical efficiency win. C0 also chooses no workspace.

All original fixed raw/workspace readouts reproduce:maximum aggregate discrepancy5.11246852e-07. All model parameter versions remain unchanged;no fallback. Native phase61.1325seconds,0.271700assigned chip-hours. Both readouts cost0.599889assigned chip-hours total. The same original30-minute absolute deadline is retained,including the failed capacity admission and intervening preparation. No fresh confirmation data was read.

The first calibration on already-optimized examples over-sharpened every continued model and worsened held-out loss. Its apparently positive workspace utility was against that degraded reader;it remains separately published and is not a win. This source-separated correction keeps the same grid and models and is also negative. No further mixer,temperature grid,length,router or larger fit is admitted by these results.

[Original uniform-NTP result](../../spliced_workspace_discovery_20261028/REPORT.md) · [First calibration](../REPORT.md) · [Source check](source-check.json) · [Complete report](report.json) · [Costs](execution-costs.json) · [Series costs](../series-costs.json)
