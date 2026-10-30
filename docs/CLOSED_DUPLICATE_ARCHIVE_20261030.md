# Preserve closed duplicate checkpoint bytes through exact archives

The owner authorizes reclaiming unneeded storage. A read-only metadata census
finds11 candidate groups among447 raw checkpoint files from successful Native
allocations closed before October27. Size/CRC similarity is only a candidate;
no raw copy can be removed on that basis.

Within these explicitly closed groups, retain one immutable raw parent and
share only payloads whose SHA256 is identical. Preserve every other byte,
including ZIP headers and pickle metadata, literally. Require reconstruction
of the complete child's original SHA256 and length before unlinking its raw
duplicate. Materialize one complete sample and verify it too. Protect any raw
file already referenced by an existing archive, and every selected new parent.
All original model, optimizer, gradient, failure and source evidence stays
recoverable byte-for-byte; no scientific result or source corpus is deleted.

Bounds:one streaming archive at a time, at most12GiB original children, each
at most2GiB,20 minutes total. Preserve512GiB host memory,35GiB output reserves,
10,000 inodes and1.5GiB for the active ordinary control's future artifacts.
Reserve worst-case literal/archive and materialized-copy space before each
file. Stop starting new files if the known scientific controller closes; do not
overlap an unknown successor. Its955 source hashes must remain unchanged.
Publish aggregate freed bytes, costs and exact-restoration evidence; private
tensor bytes and archive recipes remain private. Restore with the existing
lossless_zip_delta.restore_archive(recipe,target) before a legacy raw-file read.
