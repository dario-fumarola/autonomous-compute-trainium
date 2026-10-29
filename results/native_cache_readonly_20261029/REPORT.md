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

## Bounded CPU serialization probe

Two otherwise identical empty MLIR modules with different source locations
produce different serialized bytes and different keys under the installed
`_serialize_mlir_module` and `_content_key` functions. Re-parsing their assembly
without debug locations gives equal bytes. This reproduces location sensitivity
in the default key path. It does not establish that retained locations explain
the active fits' misses: that still needs equivalent real compiler inputs.

The [reproducer](../../scripts/repro_neff_location_key.py) executes the unchanged
functions extracted from the installed SDK in a read-only CPU container, with
no accelerator devices, two CPUs and a two-GiB limit. Two successful probes take
1.454 seconds total; an earlier host launch fails before interpreter execution
because the Native Python interpreter is container-only. No SDK, launch setting,
scientific graph or cache content is changed. This is a local candidate report,
not an upstream submission or a qualified performance improvement.
[Receipt](location-key-probe.json).

## CPU export follow-up

An actual two-operation Torch export fixture now reproduces the same effect:
identical model code compiled with two source filenames has identical eager
outputs, identical exported graphs and identical StableHLO with locations
omitted. Source filenames survive lowering; serialized bytes and NEFF keys
differ. Re-parsing without locations produces identical bytes. This strengthens
the location-sensitivity reproduction beyond the earlier empty-module fixture.
It still does not run the complete Native Dynamo pipeline or attribute a
particular campaign miss. No compiler, cache or training setting is changed.

The device-free container has two CPU threads, a 2-GiB limit and no network.
Two setup failures are retained: wrong image (no interpreter), then device
backend autoload. Explicitly disabling device autoload permits the CPU fixture.
All three attempts total approximately 7.60 seconds and zero assigned chip-time.
[Aggregate fixture receipt](fx-location-key-probe.json).
