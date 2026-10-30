# Gradient batching passes numerical, recovery and whole-update speed gates

On the qualified 51.179M tied ordinary model, batching gradient reduction and
clipping lowers warm update time by about 79% in both execution orders. This
is an execution improvement, not a predictive-quality or latent-model result.

| Trial | Whole warm seconds/update | Updates measured |
| --- | ---: | ---: |
| Original, first | 0.293359 | 100 |
| Batched, first | 0.061569 | 100 |
| Batched, second | 0.061230 | 100 |
| Original, second | 0.288414 | 100 |

The order-specific ratios are 0.209876 and 0.212299, corresponding to 4.765x
and 4.710x speedups. Each trial starts from the same model, optimizer, RNG and
natural qualification windows. Whole warm intervals include synchronization
and progress-write overhead. Compilation, first-step gradient capture, resets,
checkpoint capture and closure are excluded from those warm intervals and
charged separately. Independent reaggregation reproduces both ratios.

All independent CPU gradient, optimizer-update and moment gates pass. The
largest mature gradient-family relative error is 1.31336e-6 (0.0001313%);
maximum first-step moment discrepancy is about 0.114%, below 2%. Native first
updates agree within 8.77e-7 relative to the step. Native first gradients are
not bit-identical, with overall relative difference 2.9493e-8. Complete state
matches across 64 replicas, and a fresh process restores update103 and exactly
reproduces update104. Only the previously accepted copy fallback occurs.

Qualification uses all16 chips, 413 disposable warm-allocation updates plus
one replay update. It produces no scientific training endpoint. The warm and
replay allocations cost 1.508078 assigned chip-hours; the retained zero-update
CLI failure costs another 0.015828. The CPU audit attempt takes65.244 seconds.
Eighteen focused CPU tests pass. Both the initial failure and CLI-only recovery
retain their receipts; the recovery finishes before the original deadline.

This qualifies this ordinary-model implementation at the stated batch and
context. It does not establish latency for the latent graph, end-to-end fit
speed including audits, or scientific equivalence under arbitrary changes.
No optimization is inserted automatically into a registered scientific run.

[Full checks](report.json) · [Costs](execution-costs.json) ·
[Timing reaggregation](reaggregation.json) · [Frozen recovery](plan.json)
