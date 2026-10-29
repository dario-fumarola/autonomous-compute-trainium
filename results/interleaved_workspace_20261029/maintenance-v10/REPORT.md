# Closed-artifact preservation before replication

Two named pairs of byte-identical, successfully completed qualification
snapshots now share storage through atomic hard links. Both original paths,
all serialized bytes, ownership, permissions and original receipts remain.
This recovered **2,563,465,216 allocated bytes (2.39 GiB)** in 13.25 seconds.
Historical files are immutable; copy before any future write.

A separate exact archive preserves 4,000 old compiler temporary files before
removing their loose copies. The verified archive is 39,874,560 bytes; free
inodes increased from 22,172 to 38,171. The operation took 4.83 seconds.

No scientific checkpoint, failed history, source corpus or receipt was removed.
Both operations ran at a closed worker boundary with zero assigned chips.
Private filesystem identifiers are omitted from the public receipt.

[Deduplication receipt](deduplication.json) · [Exact archive receipt](archive.json).
