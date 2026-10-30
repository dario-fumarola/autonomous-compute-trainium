# Bound mapped-page residency during source preparation

The original index-v1 attempt exceeds its8GiB process RSS bound after20.247s.
Preserve its partial index recipe, source snapshot, plan and failure receipt.
Suppress read-ahead for scattered immutable source mappings and periodically
discard their resident pages. This changes execution only, not bytes, selection,
packing, target coverage or checks. Never discard or modify source files.

One new process retries in a separate output directory under the original
absolute deadline,8GiB process limit,512GiB host reserve and all original floors.
Bind the failed plan/receipt. Compare the completed recipe's prefix exactly to
the original retained partial recipe. No models, accelerator work or fit follow.
