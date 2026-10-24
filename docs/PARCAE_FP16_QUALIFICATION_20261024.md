# Separate Parcae FP16 execution candidate

Do not change the common-FP32 scientific comparison, queued FP32 gradient
qualification, or release-style optimizer candidate. This is an optional
performance path, with no learning adoption or fresh held-out access.

The candidate autocasts encode/refine/coda computations to FP16 while retaining
FP32 master parameters, final normalization, vocabulary projection, CE and
accumulated gradients. Apply manual loss scale8192. Unscale only after local
backward or complete DDP accumulation/reduction. The Muon optimizer's specified
BF16 polar iteration is a separate policy. No optional fused AMP metadata.

The CPU screen uses the first eight already packed natural TRAIN windows and
the original numerical-only perturbed qualification weights/states. All8,180
targets count. It checks ordinary and recurrent4-forward/2-backward execution,
with no optimizer updates. The reference FP32 port already agrees with the pinned
independent upstream model. Gate both the global gradient and separate tied-head,
body, value-embedding, and applicable recurrence-parameter gradients at2%
relativeL2, and absolute NLL discrepancy at0.003. Ten-minute CPU limit.

The measured CPU errors are0.07089% ordinary and0.06823% recurrent. Every
separate group passes; maximum group error is0.07094%. NLL differences are
−0.00003438 and+0.00000812. Static-stage AOT-eager tests also preserve FP32
masters/gradients/head, match eager mixed execution exactly, and show that
disabling the policy preserves the original FP32 function/gradients exactly.
These are numerical checks at a qualification point, not trained-model quality.

The native gate waits for the queued FP32-gradient and optimizer/recovery
qualification to pass and for all managed workers to release their chips. Use
all16 chips as two independently owned eight-chip halves,32ranks each, fixed
context1024/batch1/accumulation8 and the same256 natural TRAIN windows, supplied
states, weights and real-target normalization as the original independent CPU
FP32 references. Require unchanged weights, identical complete replica gradients,
no unexpected fallback beyond `aten::_to_copy`, and both local/global2%
gradient and0.003-NLL gates. Additionally gate each named gradient family at2%,
so a dominant embedding/head gradient cannot hide a body error.

Native capture/comparison has a30-minute total limit and latest phase start
03:00UTC October25. Waiting reserves no chips. Freeze sources, CPU screen and
reference artifacts before launch; record compilation, capture, failure and
CPU costs separately. Use ownership-scoped cleanup and existing160GiB/half,
320GiB aggregate limits,512GiB MemAvailable and35GiB disk reserves. Leave SDK,
drivers and XLA untouched. No experiment or optimizer update is authorized by
a gradient pass. Full training integration, checkpoint recovery, mature-state
precision checks and100 warm updates remain gates before a measured fit budget.
