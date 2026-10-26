# Lossless storage maintenance

The same-filesystem proof recovers 200,044,544 bytes from one completed
checkpoint using `fallocate --dig-holes --keep-size`. SHA256, logical size,
inode, owner and permissions are unchanged. This replaces only physically
allocated zero-filled blocks with filesystem holes; every checkpoint byte and
scientific state remains available. An initial unprivileged open was denied
without changing the file; the successful operation retains the original
root ownership and permissions.

A bounded pass now applies the same before/after checksum check to successfully
closed, zero-auxiliary-weight scientific checkpoints. It excludes the entire
active main campaign, stops after 20 GiB recovered or 256 GiB scanned, and has a
900-second cap. The protected shared reserves still apply. No model forward,
optimizer update or accelerator work is performed. Its CPU work overlaps
training and is not added again to native reservation cost.

[Proof](proof.json) · [Current aggregate](status.json) ·
[Exact tool](../../scripts/sparsify_completed_checkpoints.py).
