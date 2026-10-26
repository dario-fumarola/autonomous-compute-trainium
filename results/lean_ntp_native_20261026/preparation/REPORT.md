# Lean NTP: numerical preparation only

The CPU-only preparation passes original initializer/model/source checks,
private/public CPU-parity receipt identity, and deterministic full-gradient
one/two-pass table readbacks. It takes 0.494 seconds and performs zero model
forwards, optimizer updates or accelerator work. 284 CPU tests pass.

It reuses the existing qualification trainer and independent upstream checker
with an explicit primitive-attention/runtime-optimizer wrapper. The prepared
models omit the 24.608M-parameter unused auxiliary head, retaining all 141.964M
prediction parameters. The previous CPU parity result is inherited evidence,
not a new numerical result from this metadata check.

The legacy worker selector `poisson2` is supplied an all-(2,2) table here. It is
fixed two-pass full-gradient qualification; it does not sample depths. No native
job or scientific fit has been admitted. No trained qualification state may
enter science. No TEST, confirmation reserve or terminal alpha is accessed.

Before the running development outcomes, the follow-up gate is frozen: both
seed-specific gains must be positive against both work controls, and each
two-seed mean must reach 0.01 nats. A separate complete numerical qualification
and measured budget would still be required after that gate. A weak result
will not automatically buy another run.

[Protocol](../../../docs/LEAN_NTP_NATIVE_PREPARATION_20261026.md) ·
[Preparation receipt](report.json) · [Tests](cpu-tests.json).
