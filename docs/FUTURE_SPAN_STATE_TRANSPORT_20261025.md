# Explicit stable-state transport qualification

The scientific checkpoint format binds its signature to its exact plan hash.
A new work-matched continuation therefore cannot silently reuse an old state
under a different plan. The new transport helper changes only that binding and
appends provenance. It retains the original case ID, historical schedule,
source cursor, exposure/work counters, complete model/optimizer and all32 CPU
RNG states. The existing continuation validator still requires a registered
stable parent and an independently defined future cooldown.

Transport refuses changes to model, objective, data, initializer, seed, noise,
precision, depth/work ledgers or optimizer execution. It also rejects altered
optimizer history, qualification weights, annealed states and unregistered
parents. Source-file and native-receipt hashes anchor the actual parent.

A25.46-second CPU check uses both actual seed5231 one-pass checkpoints at960
updates/502,444,236 real targets. Both full serialized model/optimizer/RNG
copies restore exactly;original files remain unchanged. No model forward or
optimizer update occurs. Native launch is disabled in the qualification plan
and its communication-port inventory is empty. This is not a scientific-run
admission or a claim of identical future trajectories after restart.

The prospective2365-update control still needs native continuation/update
qualification before a separately frozen scientific registration. The copies
remain private and never replace the original stable checkpoints. No extra
chip allocation,held-out score or terminal-alpha expenditure is involved.

[Actual tensor-copy evidence](../results/future_span_transport_20261025/report.json)
· [225 passing CPU tests](../results/future_span_transport_20261025/cpu-suite.json).
