# CPU static-stage compilation qualification

AOT-eager fixed-shape encode/refine/decode matches logits and all gradients
exactly in five checked configurations. Parameter/checkpoint names are unchanged.
Final attempt:12.796s including imports, zero chips or optimizer updates.

The first helper failed while writing its report; argument shadowing is repaired
and the failure is retained. No result from it substitutes for attempt2.
Native compilation, distributed execution and throughput remain unqualified.

[Final evidence](attempt2/report.json) · [Receipt](attempt2/execution-costs.json) ·
[First failure](attempt1/failure.json).
