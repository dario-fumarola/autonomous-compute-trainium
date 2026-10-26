# Lean NTP CPU check: passes, no scientific launch

Removing the zero-weight training-only future head leaves CPU FP32 NTP losses,
prediction gradients, clipping norms and prediction parameters after the first
original-optimizer update exactly unchanged in both checked arms. Each arm uses
four original TRAIN windows from the frozen seed5231 initializer, at one or two
fully differentiated passes. No updated state is saved or adopted.

This removes24,608,000 training-only parameters, retaining141,963,840 deployed
parameters. The nominal global512-window matrix ledger becomes
270,391,480,478,208 operations/update for One and326,673,335,904,768 for fixed2.
One's counted work falls1.72865%;this is not a measured throughput improvement.

A subsequent read-only check of both stable1,004,887,590-target scientific
checkpoints finds the entire auxiliary weight,bias and fixed-position buffer
exactly equal to their original unfitted initializer. Checkpoint/signature
hashes are verified before and after access. The check takes7.60 CPU seconds,
uses zero chips and makes zero model forwards or optimizer updates. This
supports treating the head as unused capacity;it neither removes its charged
work nor qualifies future native lean trajectories.
[Trained-state check](trained-head-invariance.json).

A prospective lean One challenger to the running full-head recurrent candidate
would receive2,353 updates/1,231,507,928 targets at the1B comparison,or
4,706 updates/2,463,002,486 targets at2B. Each is0.0178874% below the candidate's
nominal work and the next update exceeds it. These are cost calculations,not
trained endpoints and not substitutions in the frozen MAIN.

The check takes67.07 CPU wall seconds,uses zero accelerator chips,and makes
four qualification-only optimizer calls. It checks an unfitted initializer,
not mature-model trajectories. Native FP16 execution,full512-window gradients,
optimizer-state recovery and actual throughput remain unqualified for a lean
scientific trainer. The live six-endpoint MAIN remains unchanged.
278 CPU tests pass after adding the checker. No held-out scoring,TEST,reserve
access or terminal alpha is used.

[Protocol](../../docs/LEAN_NTP_CPU_CHECK_20261025.md) ·
[Numerical result](report.json) ·
[Prospective cost calculation](prospective-matched-work.json).
