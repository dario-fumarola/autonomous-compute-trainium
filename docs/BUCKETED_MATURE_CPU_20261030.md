# Mature-gradient CPU qualification of communication batching

Use the four already audited billion-target gradient captures: ordinary K0,
pauses K0, and generated-state K2/K3. They use the existing independent TRAIN
audit windows. No source selection, new model scoring or accelerator execution.

For each, load two disposable CPU copies of the same complete checkpoint and
AdamW state. Compare the original per-parameter global clipping against the new
32 MiB bucket helper at gradient multipliers1 and10. Captured gradients are
already globally averaged: the helper uses world1 and a no-op reduction callback
in this CPU test. This does not qualify Native communication. Multipliers are
numerical stress fixtures, never a scientific training recipe.

Verify all complete gradients, optimizer moments and parameter updates against
the original clipping implementation. Require relative L2 errors at most1e-5;
retain every case, bucket count, norm and maximum absolute difference. The
reference and candidate optimizer states must be independently cloned. These
disposable updates never modify a saved model or initialize training. The original
Native gradient audits remain the evidence for their training histories.

One CPU attempt, ten minutes, eight threads, 8 GiB RSS, existing memory/disk/inode
floors. Preserve original input hashes and all outcomes. No fit, Native benchmark,
old-cost discount or automatic promotion is admitted. The active ordinary control
and its source manifest remain unchanged.
