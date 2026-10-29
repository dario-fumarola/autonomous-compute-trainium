# Read-only compilation-cache diagnosis

The installed `neuron_backend_fx_compiler` calls uncached StableHLO lowering.
The separate `fx_to_neff` route uses the cached wrapper. The official
[environment-variable reference](https://github.com/aws-neuron/torch-neuronx/blob/main/docs/environment_variables.md)
also limits the experimental HLO compile-cache switch to that separate API.
Enabling the switch in our launcher therefore does not establish cached
lowering for the trainer's `torch.compile` path.

An initial suspicion about ephemeral compiled-model storage was disproved:
`/tmp/neff_cache` is already a symlink into the persistent shared mount. Its
absence from the explicit environment is not a missing persistent directory.
No cache path, SDK, launcher or scientific source was changed.

The first two complete ordinary fits take 189.54 and 189.67 seconds for their
first update. This includes initialization and execution, not just compilation.
Each interval contains two last-modified NEFF files, about 8.3 MB. The reason
for these differing entries remains unisolated; no SDK defect or prospective
speedup is established. Current budgets continue to charge observed startup.

This audit only reads source, launch receipts, container paths and file metadata.
It uses zero chips, optimizer updates or held-out targets. Source hashes and
aggregate evidence are in [the receipt](report.json). No upstream submission.
