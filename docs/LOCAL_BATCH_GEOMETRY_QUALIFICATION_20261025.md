# Reduce dispatch overhead without changing the learning task

This is a bounded implementation qualification,not a new scientific model.
The current optimizer-only speedup gave less than5% whole-training improvement
and is closed. Keep the original qualified optimizer,FP16 body,FP32 primitive
attention and FP32 heads/masters/moments.

Compare local batches2 and4 at context1024 on32 ranks per eight-chip job.
Reduce accumulation from16 to8 or4 so every optimizer update still covers the
same512 original TRAIN windows. Keep initialization,source order,per-window
latent noise,global objective normalization,schedule and parameter inventory.
Only deterministic-one depth is eligible here. Later recurrent/curriculum
shapes are not admitted by this qualification.

The native driver derives from the already qualified121-update path,with
explicitly recorded batch/accumulation geometry. At each probe,`micro` names
the logical per-rank source slot0..15;additional fields record the actual
hardware microbatch and its within-batch offset. The CPU auditor independently
checks this mapping and reconstructs each window with its original noise.
Its FP32 objective and optimizer-reference arithmetic remain unchanged.

Each geometry performs121 qualification updates,including100 changing-schedule
warm updates,and repeats update121 after exact restoration of the120-update
checkpoint. Require complete32-replica equality,independent full512-window
gradients and observed-gradient optimizer checks at0/120,and replay checks.
The CPU tests compare stacked and sequential objective gradients and verify
that changing one sample cannot alter another sample's prediction.

Run both geometries on all16 idle chips under a1,800-second cap,native phases
capped at600 seconds and CPU phases900. Preserve512GiB MemAvailable and35GiB
free on output filesystems. Failures remain charged;ordinary failure cancels
only owned workers. The existing batch-one110.170-second100-update measurement
is a descriptive reference,not a swapped-hardware causal speedup estimate.
No new held-out access,corpus generation,scientific state adoption or automatic
training follows. Measure and qualify the eventual task driver separately.
