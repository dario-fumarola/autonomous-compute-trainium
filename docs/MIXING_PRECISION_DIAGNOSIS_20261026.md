# Diagnose the failed mature cheap-control gradient

The fixed-control comparison stopped because the completed seed-5231 cheap
control fails its unchanged 2% family gate: injection/readout error is 4.407%,
although overall error is 0.695% and scalar loss agrees to 0.000003814 nats.
Keep that failure and all interrupted-worker costs. Do not relax the threshold
or restart partial scientific phases under their old identities.

Use the exact failed checkpoint and original next 512 authentic TRAIN windows
for zero-update precision diagnosis. Keep weights, initial states, data order,
objective, full gradients and matrix accounting unchanged. No held-out quality
selection, optimizer update or scientific-state adoption is permitted.

Two initial native variants: `mix32` executes only the input/state adapter in
FP32, retaining the FP16 core/coda and readout projection; `full32` disables body
autocast completely. Attention and both vocabulary heads already use FP32.
If mix32 fails while full32 passes, allow one prespecified `mixc32` follow-up,
also executing the readout C projection in FP32. No additional variants.

Before native admission, check the new class against the existing full-FP32
local implementation on one actual TRAIN window at this checkpoint. All
variants with autocast disabled must preserve parameters, loss and gradients.
This is an implementation screen, not native or complete-global qualification.

Then use both eight-chip halves for the two original full-global captures.
Require exact saved source/point hashes, complete model/optimizer readback,
unchanged model weights, equal replica gradients, original routing/latent
states and only the existing fallback allowance. The independent pinned
upstream FP32 512-window objective remains the numerical reference. Export
that complete vector once; other variants may reuse it only after every point,
row, state/noise, source and count identity agrees. Preserve all original
2% body/family/head and 0.003-nat loss gates, including failures. Report named
injection/readout errors to locate the discrepancy.

Use a fixed 45-minute diagnostic clock, 600-second native phase caps and
1,200-second CPU phase caps; require complete conservative measured budget
and the standard memory/disk/inode reserves before admission. Three captures
at most, zero optimizer updates. All precision overrides must be explicit in
the new diagnostic receipt; the checkpoint's parent plan describes its trained
history and is never edited. Stop after diagnosis. Passing here does not
qualify a new training recipe, retune a rate, complete the interrupted study,
or authorize a scientific fit. No TEST, reserve access or terminal alpha.
