# Closed checkpoint copies consolidated without losing a unique state

Six completed October 23 full-state duplicate pairs were independently hashed
against their original recovery receipt. For each pair, one unchanged physical
copy remains and the other original path is now an atomic symbolic link to it.
Both original paths remain byte-readable. No unique model, optimizer state,
RNG state, source, audit, failure record or dataset was deleted.

The operation releases **13,839,130,902 bytes (12.889 GiB)**: 6,224,232,746 bytes
on NVMe and 7,614,898,156 on the root filesystem. Recorded free space afterward
is 49,805,594,624 and 47,044,206,592 bytes respectively. These six pairs no longer
provide two-filesystem redundancy; no off-instance recovery claim is made.

The first attempt encountered a directory permission error before any mutation.
It ran 117.068 seconds and released zero files/bytes. Its report's redundancy
flag describes the intended operation, not an actual change; the explicit
zero-file count and subsequent recovery admission preserve that distinction.
The permission-only recovery revalidated all complete hashes, performed and
verified the six changes, and completed in 49.023 seconds. No scientific recipe
or state bytes changed. Total CPU wall time is 166.090 seconds, zero chips.

[Failed attempt](report.json) ·
[Permission-only recovery admission](permission-recovery-admission.json) ·
[Completed recovery](permission-recovery-v1/report.json).
