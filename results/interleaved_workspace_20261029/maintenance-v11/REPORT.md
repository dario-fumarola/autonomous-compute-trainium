# Exact duplicate corpus storage

The closed failed 17B corpus build and the accepted capacity build contain
byte-identical 4,002,522,692-byte code-token arrays. Full SHA-256 checks against
the active index confirm equality before and after atomic deduplication.

Both original paths, all bytes, ownership and permissions remain. The accepted
canonical file was never opened for writing. Sharing its storage recovers
**4,002,529,280 allocated bytes (3.73 GiB)** in 11.66 seconds. The older failure
receipt and outcome remain unchanged; no source document or checkpoint was
removed. Treat both historical paths as immutable and copy before any write.

[Exact operation and identities](plan.json) · [Post-operation verification](report.json).
