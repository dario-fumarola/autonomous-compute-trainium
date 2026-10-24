# One paired3B precision-continuation diagnostic

Identity: `mature-fp32-v1`. This is a new development execution, **not** a
successful restart of the stopped `mature-fixed-v1` BF16 campaign.

Question: after a common switch to a gradient-qualified FP32 execution path at
2.010B real targets, does the fixed two-pass model retain or strengthen its
prediction advantage over ordinary training as both reach3.015B targets?

Use the preserved ordinary/fixed seed103 states at update7680. Both originate
from matched random initialization and real-source order. Their earlier
architectural interventions and BF16 histories remain part of the experiment;
this is not an all-FP32-from-initialization study or a fresh training seed.
Explicit source copies change only precision/local-batch/accumulation metadata.
Every weight, optimizer moment, RNG, cursor, counter and schedule is preserved.

Both models retain150,436,608 unique parameters, the original20 unique layers,
their ordinary/fixed1/2-pass paths, natural-only6.5B source stream, context1024,
global256 windows, original LR peaks(ordinary6e-4/fixed1e-3), AdamW(.9,.95),
matrix decay0.1, no norm/embedding decay, clipping1 and token-based WSD. The
optimizer remains the qualified unscaled fused implementation. No teacher,
synthetic research data, RL, new learned gate or larger model is introduced.

Native precision is now FP32 throughout, primitive SDPA selected before torch
import. Use32 ranks×batch1×accumulation8 per eight-chip half. The two paths
have full-global CPU FP32 gradient errors0.0001207/0.0001303, below the unchanged
2% gate. Adopt the explicitly scoped [numerical recovery policy](FP32_NUMERICAL_QUALIFICATION_20261024.md):
exact complete state load and replica equality, source/RNG preservation and
the existing2% local full-gradient probe before every phase. Future trajectories
across a process restart are not promised to be bit-identical. The failed
strict test remains a failure; its independent whole-DEV NLL difference is
0.00001295, below its registered0.0001 sensitivity limit.

Freeze stable readouts at updates8448 and10368, then complete cosine decay
from10368 through11520. At11520 both arms have3,014,637,826 real targets and
have completed their own decay. No LR changes or per-arm retuning during this
continuation. Neither arm resumes from an annealed checkpoint. Paired data
windows and exposure counters must agree at every stage.

Primary readout for this diagnostic: independent FP32 CPU NLL on the existing
603-document DEV curve panel, by web/math/code and fixed60/20/20 weights.
Native FP32 readouts must agree within0.0002 nats. Validate complete serialized
optimizer/model recovery copies as well. DEV is already consumed for development;
the evaluator refuses TEST. One paired seed supports a descriptive learning
curve, not seed-robust significance or a new confirmatory claim.

The equal-data comparison is the only completed resource contrast promised by
this bounded execution. Count the vocabulary head, backward and both core
applications. Fixed still spends1.692× estimated matmul work per equal exposure.
The properly annealed approximately5.1B ordinary control remains a separate
unlaunched stage until registered and budgeted. **No compute-efficiency win may
be inferred from the3B equal-data endpoints.** Record the different BF16/FP32
fractions and actual wall/chip costs; equal estimated matmuls do not equate
mixed-precision wall cost.

Measured concurrent100-warm-update rates are162,899 fixed and176,468 ordinary
real targets/second, with99.985% overlap of the shorter warm interval. Budget
remaining source work from these rates, plus195 seconds per training launch,
100 seconds per readout,30% margin and30 minutes of CPU tail. Training cutoff
09:40 UTC, hard cutoff10:30 UTC on October24. Refuse launch if the complete
budget no longer fits. Periodic full checkpoints every1024 updates; phase
failure stops both halves and is recorded. An incomplete phase requires an
explicit new recovery identity rather than a silent rerun.

Freeze source hashes, source-copy hashes, data/panel identities, counts and
budget before the first new update. Preserve all stopped qualification costs
and previous results. Fresh confirmation still requires newly held-out source
groups, independently replicated seeds and the ongoing charter's alpha ledger.
This diagnostic spends no terminal-test alpha and does not complete that goal.
