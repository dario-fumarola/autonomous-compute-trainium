# Global readout on already-trained examples

This diagnostic is complete but does not rescue the workspace. Its calibration used the first640 TRAIN views,which all continued models had already optimized. Every continued model chose temperature0.9 and worsened held-out NLL. A positive workspace contrast against an over-sharpened reader is not a net improvement over its original raw prediction. No backbone updates occurred.

| Model | TRAIN-chosen alpha | Temperature | Held-out full32 NLL | Original raw32 NLL |
| --- | ---: | ---: | ---: | ---: |
| c0 | 0.05 | 1.0 | 2.281988266 | 2.282111261 |
| ordinary | 0 | 0.9 | 2.290930508 | 2.262856181 |
| latent | 0.2 | 0.9 | 2.294467211 | 2.265101427 |
| constant | 0 | 0.9 | 2.294956125 | 2.265132233 |

Latent selected utility at the fitted temperature is+0.008858nats,but full32 NLL2.294467 is worse than its unmodified raw2.265101 and original fixed workspace2.267065. Ordinary also worsens to2.290931. Neither is a useful improvement. C0 choosesalpha.05 and improves by only0.000123nats full32.

Original raw/workspace endpoint scores reproduce within1e-5;all parameter versions remain unchanged,no fallback. Native execution73.842seconds,0.328189 assigned chip-hours,zero optimizer updates. One seed and reused448-view TUNE panel.

A source-reserved follow-up will use deterministic unused TRAIN groups disjoint from every backbone continuation group,under the same original absolute deadline. This is a correction for in-sample calibration bias,not an expanded grid or a replacement of this negative.

[Registration](../../docs/SPLICED_WORKSPACE_READOUT_20261028.md) · [Complete report](report.json) · [Costs](execution-costs.json)
