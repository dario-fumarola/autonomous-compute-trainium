# Prediction-error workspace: calibration gate fails

The fixed causal error correction improves the reused TRAIN calibration mean by only0.000245 nats. It does not reach0.005 and loses to the independently calibrated prediction-only control. The registered decision is to stop here: no new TUNE score, learned fit or larger campaign.

| Method | Selected step | Calibration NLL | Gain over C0 |
|---|---:|---:|---:|
| Frozen C0 | 0 | 2.326380380 | — |
| error | 0.25 | 2.326135679 | +0.000244701 |
| uniform | 4 | 2.325929424 | +0.000450956 |
| label | 0 | 2.326380380 | +0.000000000 |
| prediction | 0.25 | 2.326027985 | +0.000352395 |

Every method received the same five step sizes on the same797 calibration views/482 source groups (499web,124math,174code),with the original60/20/20 aggregate. The full grid and domain scores are retained. These are selected calibration values,not held-out generalization estimates. Neither the prediction-only nor uniform-error gain is a confirmed result.

The operation stores one negative gradient of past prefix prediction loss in a temporary vocabulary bias. Current/future targets never enter its writes. It tests a known dynamic-evaluation idea;it is not a learned thought writer or evidence of reasoning,from-scratch learning,autonomous allocation or compute efficiency. Failure of this fixed bias update does not rule out all fast-weight learners.

The first FP32 attempt fails qualification before any quality score (maximum corrected target-logp discrepancy0.000655). Recovery changes only correction normalization,accumulation and scoring toFP64;the base,source set,grid and criteria remain unchanged. All13 focused tests pass. Actual-shape error vectors match an independent literal weighted-CE gradient to8.604e-15 relative L2,with maximum target-logp error1.599e-13. Exact zero-rate raw identity and historical calibration parity pass (maximum5.258e-7). Checkpoint/head versions and all bound hashes remain unchanged.

Recovery wall time481.46seconds,including431.63seconds calibration;counted dense matrix operations54,762,507,730,944. Zero persistent model/optimizer updates and zero assigned chips. These timings include an unoptimized correction implementation and are not serving measurements. The failed attempt and shared foundation/feature costs remain counted.

The320 protected source groups,TEST and original confirmation pool remain untouched. The earlier associative-cache and supported-iteration negatives remain unchanged. The broader workspace goal is unachieved.

[Recovery aggregates](report.json) · [Full calibration grid](calibration.json) · [Numerical audit](numerical-audit.json) · [Protocol](../../../docs/PREFIX_ERROR_HEADROOM_20261028.md)
