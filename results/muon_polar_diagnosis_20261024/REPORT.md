# Muon polar precision: same-runtime CPU, eager and compiled

Eight independent numerical cases use four matrix shapes—768x768,3072x768,
768x3072 and6x32—with the released BF16 polar arithmetic or a diagnostic FP32
variant. CPU and native use the same Torch2.14 runtime, inputs and operation
sequence. Arrays are software fixtures, never research examples. No optimizer
or model training update ran; the installed SDK is unchanged.

| Polar dtype | Native eager final error vs CPU | Native compiled final error vs CPU |
| --- | ---: | ---: |
| BF16, three large matrices | 1.47–1.51% | 4.87–4.92% |
| BF16,6x32 gate | 0 | 12.37% |
| FP32, all four shapes | 0.00018–0.01097% | 0.00019–0.01097% |

Errors are relativeL2 on the fifth polar output, not parameter updates or
training quality. BF16 eager normalization denominators exactly match CPU in
these fixtures; compiled denominators differ by0.407–0.478%. Differences grow
through the subsequent polynomial iterations. The trace establishes a compiled
versus eager numerical distinction; it does not yet identify a compiler pass or
justify treating every BF16 discrepancy as an operator-contract violation.

No fallback is reported. Worker time27.006s,0.060014 assigned chip-hours;
controller33.678s. FP32 is a diagnostic, not an adopted replacement optimizer.
The next bounded check tests the complete actual-shape eager BF16 update against
the unchanged independent CPU vectors before any scientific recipe changes.

[Checkpoint-free reproducer](../../scripts/repro_muon_polar_precision.py) ·
[Exact stage errors](report.json) · [Frozen plan](plan.json) ·
[Costs](execution-costs.json).
