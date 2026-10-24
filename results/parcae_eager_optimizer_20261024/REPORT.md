# Eager BF16 Muon: improved, still fails the fixed gate

Changing only Muon execution from compiled to eager reduces actual-shape update
error to1.0462%,0.9436% and2.3205% across the three prescribed steps. The signed
third step still exceeds the unchanged2% per-family gate, so the screen fails.
Complete Muon-moment errors are below0.00521%; AdamW remains below6.14e-6.
All32 complete replica states agree. No replay or scientific training follows.

Worker time78.931s, assigned0.175402 chip-hours; controller including CPU
comparisons134.024s. The two warm prescribed optimizer calls take about0.06s
each, but these are not full-training throughput measurements. They also show
that compilation need not be the faster execution for this small optimizer.

No gate is relaxed. The separate prospective correction uses **FP32 polar
arithmetic**, an explicitly changed optimizer precision recipe, with its own
independent modified-upstream CPU vectors and native/recovery/profile gates.
This report does not adopt it, change a scientific model, or erase either BF16
failure. [Exact comparisons](report.json) · [Plan](plan.json) ·
[Costs](execution-costs.json).
