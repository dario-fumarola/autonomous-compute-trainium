# Recurrent0.016 TUNE cell stopped as ineligible

The trial completes62 finite updates /32,449,078 real targets, then raises `Nonfinite global sharded gradient; no update` before the next optimizer mutation. The supervisor reports zero OOM and no reserve/compiler failure. No held-out score exists for this cell. It remains ineligible and all its costs stay counted.

The short TUNE path saved only299/300 endpoints, so the exact live failing model/optimizer/RNG state was not captured. The source,C0,all62 update receipts and error/closure evidence remain. A reconstructed trajectory would be labeled as reproduction, never the original state. Fixed manual-scale overflow and actual training divergence are still alternatives; no causal diagnosis is claimed.

Before any further cell, add a failure-only wrapper that leaves every successful update unchanged and captures every raw local model/gradient/optimizer/RNG shard before rethrowing the same nonfinite exception. An injected CPU transport fixture verifies preservation,no model update,ineligible tagging and error propagation; the real64-rank error path remains unobserved until an actual event. Do not relax gradient limits or import invalid states into science.
