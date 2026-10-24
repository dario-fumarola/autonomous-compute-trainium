# Conditional native FP16 gradient qualification

**Suspended before any native phase; zero chips reserved.** No native result is established.

The preceding optimizer implementation fails the separate early Muon update
screen. This unstarted queue is suspended while that arithmetic is diagnosed;
all original frozen sources remain unchanged. The original requirements below
remain unfulfilled. [Suspension](early-screen-suspension.json) ·
[Early optimizer evidence](../parcae_early_optimizer_20261024/REPORT.md).

Prerequisites are a passing full native FP32 gradient gate, passing native
optimizer/recovery gate, and no existing managed native workers. The current
scientific comparison retains FP32 and finishes first. Both ordinary/recurrent
captures then use eight chips each, context1024/batch1/accumulation8, and exactly
the independent256-window natural TRAIN objectives, initial states and weights.
No optimizer update, held-out scoring or terminal alpha.

Manual loss scale8192 is removed from FP32 gradients after complete DDP
accumulation/reduction. The final norm/head/CE remain FP32. Require local and
global gradient agreement within2% overall and in each named family, NLL within
0.003, identical replica gradients, unchanged weights and no unexpected fallback.

Total native execution/capture/comparison limit30 minutes; latest phase start
03:00UTC October25. All source files and reference artifacts freeze before
launch. CPU success does not establish native behavior; a native gradient pass
does not establish optimizer integration, full trainer recovery, mature-state
precision or100 warm training updates. No scientific fit follows automatically.

[Registered plan](plan.json) ·
[Full protocol](../../docs/PARCAE_FP16_QUALIFICATION_20261024.md) ·
[CPU screen](../parcae_cpu_fp16_20261024/REPORT.md).
