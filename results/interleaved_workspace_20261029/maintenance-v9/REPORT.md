# Exact closed-snapshot deduplication

Two pairs of old owner/replicated qualification snapshots have identical
original recorded hashes and verified current serialized bytes. Atomic hard
links now share those duplicate blocks while retaining both paths, ownership,
permissions and exact contents. Allocated space recovered:2,971,430,912bytes.
Differing full-state and gradient files remain unchanged. The unfavorable
throughput result, all receipts and both histories remain available.

These closed historical files are immutable; copy before any future write.
No scientific state or failure history is deleted. The action takes15.77s,
zero chips, after the deeper comparison closes and before wider qualification.
[Verified receipt](deduplication.json).
