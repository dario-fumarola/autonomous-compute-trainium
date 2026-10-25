# Fixed-two small-core NTP: complete bounded tuning

All three301,465,908-target trials complete their cooldown. Three counted
121-update prefixes and the selected final checkpoint pass independent complete
512-window FP32 gradient/loss gates. Full model/optimizer/RNG loads and complete
replica checks pass. No rate or scientific case was dropped.

| Peak LR | Independent TUNE NLL |
| --- | ---: | ---: |
| 0.008 | 2.984628840 |
| 0.016 | 2.926083633 |
| 0.032 | 2.902721105 |

Select **0.032**,the upper boundary of the original three-rate grid. Historical
One/NTP selected the same rate and scores**2.938676632** on this panel. The
fixed-two winner's equal-data selection gain is**0.035955527 nats**. It spends
more training work. This is adaptive TUNE selection,not a held-out MAIN or
compute-efficiency result. Both optima remain unbracketed;no rates are added.

The same qualified operators,model,optimizer,batch and raw-source order are
retained;only the exogenous depth table is fixed at exactly two fully trained
passes. The zero-weight common future head is retained and counted in training,
removed at scoring. The actual qualified cosine schedule ends at zero;no
executed schedule was changed to repair a description.

**1,728 actual new research updates**,904,397,724 target exposures across
three LR trials,with301,465,908 common distinct source positions per trial.
Measured100-update intervals take132.27–135.38s. The selected full512-window
endpoint gradient discrepancy is**0.30184%**,below the2% body/family gate.
All10 native phases exit cleanly. Assigned native cost**7.6072 chip-hours**;
whole-instance opportunity**15.5862 chip-hours**,separately/nonadditively,
including CPU tails. Historical One tuning receipts remain separately preserved.

The separately registered finite MAIN is now launched from the same original
random initializer for both arms,with1B/2B equal-data and exact One-work endpoints.
No TEST,reserve,terminal alpha or terminal success.278 CPU tests pass.
[MAIN protocol](../../../docs/FIXED_TWO_SMALL_CORE_MAIN_ADMISSION_20261025.md) ·
[Exact tuning](report.json) · [Receipts](execution-costs.json).
