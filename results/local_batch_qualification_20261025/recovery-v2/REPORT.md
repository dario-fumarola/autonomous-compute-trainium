# Local-batch qualification

Both geometries pass all six independent complete512-window CPU FP32 gradient/
optimizer checks,with maximum body relative-L2 error0.09010%. Each update still
uses512 original windows on32 ranks. All replicas agree,and full model,
optimizer and per-rank RNG state restore exactly. **Future-step state digests
differ across restart**;numerical replay passes,but bit-exact future trajectories
are not established.

| Local batch | Accumulation |100 warm updates,seconds|Targets/s per8-chip job|
|---|---|---:|---:|
| 2 | 8 | 82.993 | 630627 |
| 4 | 4 | 85.927 | 609092 |

The previous local-batch-one measurement was110.170s/100updates. This is a
cross-launch descriptive reference,not a hardware-swapped causal comparison.
Batch2 is the preferred qualified deterministic-one geometry. No recurrent or
curriculum objective/shape is admitted by this gate;the eventual driver still
needs its own numerical and recovery checks.

The failed first attempt used absolute links that were invalid inside the
container. Recovery uses portable relative asset links and retains the original
absolute deadline. Both cost receipts remain published;recovery's opportunity
interval includes original admission and overlaps the failed attempt,so those
opportunity costs must not be added.244 native qualification updates,zero
research updates. No TEST or terminal alpha.
