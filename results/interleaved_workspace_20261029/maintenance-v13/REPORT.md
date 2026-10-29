# Exact split archive of closed failed-build source arrays

The stopped17B preparation failed because its code target was undersupplied.
Its old math and web arrays consist exactly of the accepted canonical arrays
plus suffixes. A read-only full-hash audit established that relationship first.

The archive retains independent hard links to both accepted prefixes and writes
every unique suffix byte. Reconstructed full streams match the old complete
SHA256 digests before their loose failed-build arrays are retired. The archive
therefore remains restorable even if canonical paths are later renamed. Those
hard-linked prefixes are immutable; copy before any future modification.

This recovers16,009,969,664filesystem bytes (about14.91GiB), taking62.41seconds
with zero chips. The old loose math/web paths are now absent. Their failure,
source metadata and exact reconstruction information remain. No accepted corpus,
source index, scientific checkpoint or unique source byte was discarded.

Use the tested repository restore command on one domain at a time:

```sh
sudo python3 scripts/archive_closed_corpus_prefixes.py --root "$ROOT" --action restore --domain math
sudo python3 scripts/archive_closed_corpus_prefixes.py --root "$ROOT" --action restore --domain web
```

Restoration requires full file capacity above the protected reserve, checks the
complete archived hash, and refuses to overwrite an existing array. Private
archive manifests preserve original file modes, ownership and modification time.
Three software fixtures cover restoration independent of canonical paths and
fail-closed behavior on changed source/archive bytes.

[Read-only audit](read-only-audit.json) · [Archive receipt](archive.json).

A separate closed-worker exact archive preserves another4,000 old compiler files
and12,000 directories. It restores15,999free inodes in5.09seconds, with no
scientific states changed. [Compiler receipt](compiler-archive.json).
