# Fixed-position cache qualification

The slot-unbind candidate fails its registered execution screen: 4.487 seconds
per update, with cold independent gradients passing. Preserve that negative.
The device trace confirms approximately 4.137 seconds inside the forward NEFF,
versus 0.054 seconds in its backward NEFF. CPU copies wait for queued work;
their long duration is not evidence that copying causes the slowdown.

The next implementation-only test precomputes the two repeated rotary caches
on CPU at model construction. These are deterministic nonpersistent buffers,
not parameters or learned states. Forward source/latent executions slice the
appropriate cache instead of gathering rotary values using device indices.
The original strided state extraction is retained to isolate this change.
CPU outputs and every gradient are bit-identical across ordinary, pause and
several latent depths/lengths. The exact sequential reference is unchanged.

Use the same 51.75M actual shape, seed, natural software-qualification windows,
FP32 precision, optimizer, loss and matrix-work ledger. All 16 chips perform
cold CPU agreement, 100 warm updates, all-replica checks, independent mature
global gradients and exact separate-process replay. Stop at update 12 if nine
recent updates average above 0.8 seconds. Disposable weights only; no quality
readout or scientific training admission. The original failed runs retain
their identities, arrays, measurements and costs.
