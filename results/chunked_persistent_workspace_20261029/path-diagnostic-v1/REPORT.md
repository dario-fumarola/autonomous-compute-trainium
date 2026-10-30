# Sparse workspace path intervention

The saved sparse checkpoints show no measurable benefit from persistent latent memory. Reading the four boundary predictions from their original source states improves loss, but does not close the gap to the independently trained ordinary model.

| Family | Readout / memory | Broad NLL | Web | Math | Code |
|---|---|---:|---:|---:|---:|
| generated | authentic | 3.292243358 | 3.793067821 | 2.964857459 | 2.117155866 |
| generated | source_boundary | 3.291340777 | 3.792053123 | 2.964280689 | 2.116263827 |
| generated | blocked_memory | 3.292269227 | 3.793125422 | 2.964833868 | 2.117135999 |
| generated | both | 3.291367894 | 3.792111519 | 2.964257841 | 2.116247073 |
| constant | authentic | 3.292884560 | 3.795751204 | 2.961359972 | 2.115809219 |
| constant | source_boundary | 3.291303591 | 3.793947844 | 2.959795340 | 2.114879086 |
| constant | blocked_memory | 3.292918913 | 3.795756053 | 2.961458389 | 2.115868015 |
| constant | both | 3.291336309 | 3.793951573 | 2.959894733 | 2.114932092 |

Positive differences below mean that removing the path improves NLL. The intervals are conditional, unadjusted source-group bootstrap intervals on reused DEV, not training-seed uncertainty.

| Family / intervention | NLL improvement | Conditional 95% interval |
|---|---:|---|
| generated-source_boundary | +0.000902581 | [+0.000656590, +0.001157193] |
| generated-blocked_memory | -0.000025869 | [-0.000075354, +0.000023327] |
| generated-both | +0.000875463 | [+0.000616479, +0.001139344] |
| constant-source_boundary | +0.001580969 | [+0.001251838, +0.001955517] |
| constant-blocked_memory | -0.000034353 | [-0.000083140, +0.000016275] |
| constant-both | +0.001548252 | [+0.001218250, +0.001926095] |

The generated model's memory removal changes loss by only −0.000025869 nats, with an interval spanning zero. The analogous constant-input intervention is also inconclusive. These results do not support memory contamination as the cause of the deficit, nor establish useful persistent computation.

The generated boundary penalty is 0.115530 nats per affected target, or 0.000903 across all targets. Even removing both paths leaves NLL at 3.291368 versus 3.287681 for independently trained ordinary. Training-history differences remain; this is not a repaired trained recipe or a new efficiency comparison.

Both interventions preserve all source positions and still execute the slots. Blocking memory recomputes every affected state; the current slot can still read itself and all source keys. The combined intervention exactly matches these same weights' ordinary path. Such path removals are unsupported inference interventions, not trained zero-workspace exits.

All 384 original source groups and both selected checkpoints are included. All 96 identity checks pass (maximum log-probability discrepancy 3.56e-14). Authentic domain and aggregate scores reconstruct the completed comparison. Every position-bin and domain result, and the two-factor interaction, remain in [the aggregate report](report.json).

Cost: 254.449 CPU wall-seconds, zero assigned chips, zero optimizer updates. Both fresh reserves remain unscored; terminal alpha remains zero. The original negative result and unmet advancement gate are unchanged.

[Frozen diagnostic](../../../docs/CHUNKED_PATH_DIAGNOSTIC_20261030.md).
