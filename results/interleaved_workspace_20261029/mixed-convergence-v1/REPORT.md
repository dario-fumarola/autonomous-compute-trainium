# Mixed-round sequential consistency: complete

All three mixedK2/K3 endpoints pass the unchanged six-TRAIN-window consistency guard after their mature K2/K3 gradient and full-state audits. No held-out score is used to choose these checks.

| LR | Domain | Absolute mean NLL shift | Forward KL | Pass |
|---|---|---:|---:|---|
|0.0003|web|0.00577549|0.00493907|True|
|0.0003|math|0.00805977|0.00908033|True|
|0.0003|code|0.00598806|0.00645289|True|
|0.001|web|0.00390099|0.00615159|True|
|0.001|math|0.00694723|0.00622342|True|
|0.001|code|0.00690188|0.00342376|True|
|0.003|web|0.00106378|0.00167734|True|
|0.003|math|0.00149458|0.00281659|True|
|0.003|code|0.00007107|0.00112920|True|

Each domain must retain absolute mean loss shift and forward KL at most0.01; K16/K32 KL must be at most1e-4. All requirements are unchanged. FixedK2 training failed at the first two learning rates; mixed training passes at those same settings, with more counted training work.

This is evidence for numerical consistency on the registered TRAIN diagnostic. It does not by itself establish predictive quality, full-DEV sequential agreement, reasoning or efficiency. Common held-out LR selection is still separate. A selected-checkpoint exact sequential broad readout is prepared and requires the complete predictive gate.

All3 endpoints, original failures and costs remain recorded. No larger fit or confirmation is admitted by this report. [Complete receipt](report.json) · [Frozen plan](plan.json) · [Original consistency report](../convergence-v1/REPORT.md).
