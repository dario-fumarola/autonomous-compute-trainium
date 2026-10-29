# Generated states retain a predictive benefit under sequential execution

The preselected latent checkpoint scores **3.340201 NLL** when its continuous
states are generated sequentially from the available prefix. It improves
**0.036910nats** over the equally exposed ordinary model and **0.050927nats**
over trained constant pauses. Every domain favors generated states in both
comparisons. No weights were changed or additional candidate selected.

| Predictor | Web | Math | Code | Weighted60/20/20 |
|---|---:|---:|---:|---:|
| Ordinary |3.869100|3.049119|2.229139|3.377111|
| Trained constant pauses |3.895149|3.053362|2.216832|3.391128|
| Generated states, exact sequential |**3.829902**|**3.013348**|**2.197951**|**3.340201**|
| Same latent weights, finiteK2 |3.778768|2.945996|2.124244|3.281309|

The conditional source-group95% intervals are[0.031871,0.041938] against
ordinary and[0.045774,0.056287] against pauses. These describe the selected
trained weights and reused384-group DEV panel. They do not account for seed
variation, learning-rate selection or adaptive research. Both fresh reserves
remain closed. This is development evidence, not confirmation.

Sequential generation loses0.058892nats relative to the finiteK2 readout. Thus
the learning-to-execution mismatch explains much, but not all, of the apparent
equal-data advantage. It remains a material limitation; the original
consistency guard is not relaxed. Mixed-round training is a separately
qualified and registered follow-up.

All models began from paired random initialization, received128,025,288 authentic
targets and used uniform natural-token NTP. No teacher, generated source corpus,
reasoning labels, policy reward or target-aware selection was used. The learned
continuous state is passed into the next causal position and retained in
subsequent attention history. The trained pause control is useful evidence
against explaining the gain solely by adding constant positions, but it does
not establish transferable problem solving or autonomous compute allocation.

Generated training used4.937times ordinary nominal matrix work. There is no
matched-training-work result for this mechanism yet. Sequential generated and
pause paths both process one additional state position per source token, but
their scheduling and latency differ. This CPU readout is not an accelerator
serving benchmark or a hardware-time efficiency claim.

Four CPU workers scored all196,608 frozen targets in738.90seconds. Each used
FP64, with no optimizer updates or accelerator allocation. Six independent
cached-versus-full-prefix checks agree to maximum3.56e-15nats per target.
Comparators retain their already qualified Native scores (maximum independent
FP64 error3.94e-5). Private per-view records remain outside Git.

[Aggregate receipt](report.json) ·
[Prospective registration](../../../docs/INTERLEAVED_SEQUENTIAL_DEV_READOUT_20261029.md) ·
[Original equally budgeted LR grids](../tuning-v1/REPORT.md).
