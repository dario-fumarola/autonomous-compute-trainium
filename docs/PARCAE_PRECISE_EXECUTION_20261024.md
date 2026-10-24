# Explicit FP32-polar qualification amendment

The compiled and eager BF16 optimizer screens fail their unchanged2% actual
Muon-update gate. Preserve those results and every original frozen source.
No scientific fit used that optimizer. This new execution changes **only the
polar arithmetic dtype to FP32**, identically for all prospective model arms.
Grouping, coefficients, momentum, variance normalization, cautious decay and
AdamW settings remain unchanged. It is an explicit numerical recipe deviation,
not an exact Parcae release reproduction. No accuracy threshold is weakened.

The independent CPU reference is pinned upstream code with exactly one audited
AST substitution, g.bfloat16() to g.float(), plus decorator removal for eager
reference execution. The candidate is a separate tensor implementation; the
vendored source and installed SDK stay untouched. CPU vector gates remain
2e-5 relative to update size and2e-6 on complete moments at three actual-shape
prescribed signed/scaled updates, both ordinary and recurrent. Preserve complete
expected vectors privately, publish hashes/aggregates, twenty-minute CPU cap.

Use eager native FP32 Muon and fused unscaled AdamW. This execution choice is
made before scientific outcomes: the earlier BF16 optimizer fixture's two warm
eager calls take about0.06s, versus1.3–1.5s compiled. Those are optimizer-only
measurements; new FP32 timing and complete training throughput remain unqualified.
The same-runtime FP32 polar trace agrees closely for eager and compiled modes.

New native optimizer identity `parcae-native-opt-precise-v1` freezes all three
completed scientific native endpoint receipts and waits for existing native
owners to release their chips. It may overlap the remaining independent CPU
endpoint audit: prescribed CPU optimizer vectors do not depend on native
backward. Full training still requires both independent native gradient and
optimizer gates. No active scientific source or admission policy is changed.
Use both eight-chip halves,32 replicas each. Run three prescribed updates,
save complete state after two, and replay the third in fresh processes. Require
exact complete state on load, identical complete replicas, update and moment
errors≤2% overall **and separately for AdamW and Muon**, and resumed versus
uninterrupted update error≤2e-4. No unexpected fallback beyond to_copy.
Bound the whole native optimizer execution/comparisons to30minutes, latest
phase start02:00UTC October25. Waiting reserves zero chips.

Native FP16-gradient revision2 changes only its prerequisite to the new
FP32-polar optimizer gate. Model arithmetic, fixed256-window CPU references,
manual8192 scale, FP32 head/masters,2% gradient and0.003 NLL gates,30minute cap
and03:00UTC latest phase start remain unchanged. It makes zero optimizer updates.
The suspended revision1 remains preserved and never launched natively.

Low-depth training-engine revision2 reuses the exact existing three-arm CPU
initializations, source/depth/noise plans and accumulation cases. Its only
optimizer change is eager FP32 polar arithmetic, with an explicit checkpoint
recipe/signature and source freeze. The independent first-actual-update audit
uses the one-dtype-modified upstream optimizer; the trained full-objective
gradient audit still uses untouched pinned upstream model code. Every other
gradient, recovery, source, reserve and100-warm-update gate in the original
training qualification remains. Ninety-minute complete execution cap,30minutes
per native phase/CPU audit,04:00UTC latest phase start. FP16 is included only
after its numerical pass; otherwise qualify FP32 alone. No scientific launch
automatically follows any software or performance pass.

All workers retain ownership-scoped cancellation,160GiB each/320GiB aggregate
host caps,512GiB MemAvailable and35GiB filesystem reserves. Count CPU preparation,
all warm/replayed/prescribed updates, cold compilation, failures, checkpointing,
CPU audit tails and actual chip-hours. Original source freezes are unchanged.
