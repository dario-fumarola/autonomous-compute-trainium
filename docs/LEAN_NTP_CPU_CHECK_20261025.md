# Remove the unused auxiliary head: CPU prerequisite only

The frozen fixed-two MAIN retains the zero-weight future head to preserve its
qualified reference recipe. That head is training-only and is already removed
for independent prediction scoring. Its computations remain charged to MAIN.
Do not change the active trainer or its frozen source hashes.

Before considering a lean one-pass challenger, check that removing this head
preserves the intended NTP learning problem. Use the original unfitted seed5231
initializer and the first four original TRAIN windows, unchanged targets and
exogenous state noise. Compare full versus lean primitive-attention models on
CPU FP32 at exactly one and two fully differentiated passes. Require identical
NTP losses and prediction gradients, zero auxiliary gradients, agreement of
the global clipping norm and prediction parameters after one original-optimizer
update. Save no updated model or optimizer state and adopt none into science.

`scripts/check_lean_ntp_cpu.py` also records the lean nominal matrix ledger at
the existing global512/32-rank geometry. This is a cost calculation, not a
measured speedup. It does not qualify native execution, FP16, the full global
objective, optimizer restoration or a changed local batch. A later scientific
challenger would need those gates, an independent completed cooldown, a new
matched-work endpoint and its own fixed measured budget. No launch follows
automatically from this CPU result.

Use zero accelerator chips and a900-second CPU cap, keeping existing memory
and disk reserves. Record failed attempts. No held-out panel or confirmation
reserve is opened, no scientific update is added, and no quality outcome is
selected. The current six-endpoint MAIN continues unchanged.
