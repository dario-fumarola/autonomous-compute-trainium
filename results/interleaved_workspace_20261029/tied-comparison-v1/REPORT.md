# Tied ordinary challenger: complete equal-data comparison

The exact-sequential mixed latent candidate scores **3.262734 NLL**, versus
**3.287781** for the independently tuned eleven-layer tied-table ordinary
challenger. The paired development gain is **0.025047 nats**, with conditional
95% source-group interval **[0.020119, 0.029968]**. All three domains favor the
latent candidate. The registered equal-data development screen passes.

| Domain | Latent | Tied ordinary | Gain | Conditional 95% interval |
|---|---:|---:|---:|---|
| web | 3.762500 | 3.789349 | 0.026848 | [0.019864, 0.033886] |
| math | 2.928660 | 2.957114 | 0.028453 | [0.019758, 0.038734] |
| code | 2.097509 | 2.113745 | 0.016237 | [0.007375, 0.024806] |

Both models see 128,025,288 authentic targets from random initialization, using
seed9901 and the same source order. The candidate has51,747,328 parameters;
the challenger has51,179,008. Three independent LR opportunities select the
interior0.001 for both, with complete cooldowns and mature numerical audits.
No qualification weights enter scientific initialization. The candidate score
uses exact sequential deployment, not its finite-round training approximation.

The candidate uses **3.782219 times** this challenger's nominal training matrix
work. Retained Native fit paths take790.642s versus656.963s (startup, compilation,
audit and checkpoint included). These are descriptive cross-launch times,
not matched hardware-time evidence. Analytic cached512-position inference is
62.082 versus55.354 billion matrix FLOPs, with25,165,824 versus23,068,672 FP32 KV
bytes; serving latency and throughput are unmeasured.

This is one selected seed on repeatedly inspected DEV. Its source-group
interval does not measure training-seed uncertainty or remove adaptive
selection bias. The two other pre-inventoried ordinary shapes remain to be
tested. No full-frontier, training-efficiency, general-reasoning, allocation or
confirmation claim follows. Both fresh reserves remain unscored.

[Full paired receipt](report.json) ·
[Control tuning](../tied-ordinary-tuning-v1/REPORT.md).
