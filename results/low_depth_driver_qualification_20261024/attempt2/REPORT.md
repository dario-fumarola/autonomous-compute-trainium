# Finite scientific-driver integration

Pass. Ordinary and sampled low-depth models each execute50 actual updates:
fresh20, exact full-state resume to40, and an independent branch from stable20
to30. Both endpoints finish their own cooldown to zero. All32 model/optimizer
replicas agree; restored complete state and CPU RNG match exactly. Source cursors,
real-target/domain exposure and integer executed-work counters agree throughout.

Wall time522.060seconds. These100 qualification updates use separate seed1009
and cannot initialize or resume scientific runs. No held-out data was scored.

The preserved first attempt failed before any update at communication setup.
The SDK's free-port discovery releases a candidate socket before the runtime
binds it; the exact competing socket owner was not identified. The new managed
execution registers distinct phase-specific ports outside host ephemeral
allocation and verifies their identity in each worker. This resolved the observed
collision without SDK, driver, network or sysctl changes. It does not provide
atomic exclusion against unrelated unmanaged applications.

[Exact results](report.json) · [Costs](execution-costs.json) ·
[Port policy](../../../docs/NATIVE_COMM_PORT_POLICY_20261024.md).
