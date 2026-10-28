# Short-prefix associative memory does not clear the useful-gain threshold

All three fresh 1,536-update fits and mature numerical/recovery checks complete. The fitting set contains 6,144 views from 3,393 source groups; 797 calibration views from 482 separate groups select the fixed-cache baseline. The 448-view development panel is unchanged.

| Predictor | NLL | Gain over frozen C0 |
| --- | ---: | ---: |
| C0 / calibration-selected fixed cache | 2.282112 | 0 |
| Trained frequency-cache gate | 2.281886 | 0.000226 |
| Trained frozen-cosine gate | 2.280980 | 0.001132 |
| Learned addressing and gate | 2.281840 | 0.000272 |
| Learned addressing with rotated values | 2.282398 | −0.000286 |
| Historical independent one-step residual | 2.275685 | 0.006427 |

Calibration selects zero fixed-cache weight; all nonzero combinations in the registered grid worsen its mean. Learned addressing does not beat the trained cosine control. Its gain over C0 has conditional 95% interval [−0.000097, 0.000633]. Correct key/value association helps by 0.000558, [0.000246, 0.000903], but neither the absolute nor the stronger-control/content thresholds pass. No learned-thinking result is claimed.

The strongest domain signal is the cosine gate on code: +0.005135 nats, versus −0.000063 on web and +0.000715 on math. All domains and intervals are retained in [report.json](report.json). Repeated targets have raw NLL 1.152874; unseen-in-prefix targets have 3.531147. Cosine memory improves the former by 0.003708 and worsens the latter by 0.001729. These descriptive strata are token-weighted, not the primary 60/20/20 mixture. The mechanism can only directly boost observed token identities; this is contextual copying, not computation of new answers.

Eight focused tests and full-size three-domain FP32 versus independent sequential FP64 gradients pass. Maximum mature relative error is 1.806e-6. Current/future outcomes cannot affect memory predictions; empty memory and the initial zero-strength mixture reproduce C0. All projected AdamW/RNG states restore and disposable updates replay exactly. No reserved source was scored, and no model receives a current target as a gate input.

The frozen head cache takes 132.27 CPU seconds; bag/cosine/learned fitting takes 7.22/10.22/13.18 seconds concurrently. Total controller time is 188.23 seconds. All frozen-head forwards, branch work, inherited feature/base costs and reserved-instance opportunity are recorded in [the ledger](complete-costs.json). No new accelerator or backbone updates occur. Short fitting speed does not imply a total-compute or inference-efficiency win.

This is one development seed and a reused panel. The outcome does not justify scaling this reader. The limited context span is an unresolved experimental condition: a separate frozen diagnostic on existing authentic dependencies can test longer memory without another fit, new source selection, or changing this negative result.
