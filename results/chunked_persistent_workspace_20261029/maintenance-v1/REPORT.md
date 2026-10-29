# Closed feature caches retired

Four float16 hidden-feature caches from the completed two-seed routing probe
were removed, recovering 3,941,660,912 bytes (3.671 GiB). Their bytes were checked
against the original recorded hashes before removal. Full original checkpoints
were independently verified from their exact compressed archives, and the
original packed-source manifest still matches.

Original results, source data, loss labels, domain labels, plans and checkpoint
archives are retained. This is retirement of reproducible derived caches, not
lossless archival of those four arrays. Recreating them requires the historical
code and inputs; this maintenance did not test bit-exact regeneration. No
scientific weights, scores or active execution inputs were changed. The owner
explicitly authorized removal of unneeded old artifacts.

The operation took 28.403 seconds on CPU, used no chips and performed no model
forward or optimizer update. [Receipt](report.json).
