# Trained-model numerical diagnosis

The mature BF16 fits stopped at the existing local gradient guard. This report
preserves that failure and investigates it without optimizer updates or TEST.
These are numerical measurements, not a replacement scientific endpoint.

## Full global objective

All measurements use the preserved next256 natural TRAIN windows and their real
target normalization. Every native replica has the same global gradient.

| Trained checkpoint | Native BF16 vs CPU FP32 relativeL2 | CPU BF16 vs CPU FP32 |
| --- | ---: | ---: |
| Fixed seed103, update7680 /2.010B targets | 0.209251 | 0.085418 |
| Ordinary seed103, update9216 /2.412B targets | 0.203326 | 0.049561 |

The prospective global qualification gate is relativeL2≤0.02 and absolute loss
error≤0.003. Both BF16 captures fail the gradient gate. Their scalar loss
agreement does not establish backward numerical agreement.

The repeated fixed native capture is bit-identical. Separately reducing each
microbatch and summing on CPU differs from ordinary accumulation by only
1.12e-7 relativeL2. The tested accumulation/reduction route therefore does not
explain the discrepancy. Changing the local batch from4 to1, preserving all256
global windows, changes the BF16 gradient by0.000972 relativeL2.

## Attention execution control

The fixed checkpoint, full global objective and32×1×8 source layout are held
constant. Only native precision and the SDK's NKI SDPA selection change.

| Native execution | Gradient error vs CPU FP32 | Loss difference |
| --- | ---: | ---: |
| FP32, NKI SDPA enabled | 0.029312 | +0.00005441 |
| FP32, primitive SDPA | **0.00012068** | -0.00000077 |
| BF16, NKI SDPA enabled | 0.209 approximately | +0.00024 approximately |
| BF16, primitive SDPA | 0.208629 | +0.00020180 |

The primitive FP32 route clears the unchanged global gate. This localizes the
FP32 difference to attention execution, without establishing the precise
kernel operation responsible. It does not solve the larger BF16 difference.
The NKI forward kernel's installed documentation explicitly says it uses BF16
for operations outside softmax even when accepting FP32 input. A public,
checkpoint-free reproducer is being qualified separately. Source inspection
alone is not a reproduced operator defect.

Four isolated numerical fixtures also compare primitive native RMSNorm,
linear projection, SwiGLU and causal GQA SDPA with CPU FP32/BF16/FP64. These use
seeded arrays and trained layer0 weights, never research examples. Their
results and the failed harness predecessor are recorded separately. The
harness initially expected a fused serialized QKV key; the exporter saves
Q/K/V separately. That harness mistake did not affect any scientific fit.

## Next gate

A separate FP32 primitive-attention qualification checks actual updates,
100 warm steps, complete optimizer recovery and a fresh-process continuation.
Its explicit source copy changes only precision/local batch/accumulation
metadata; all weights, optimizer moments, RNG, source counters and schedules
remain exact. Those qualification states do not replace the stopped BF16 fits.
No longer scientific run is authorized by a forward-only check.

The prior1B results remain measured comparisons of their fitted models. This
diagnosis limits claims that those trajectories reproduce higher-precision
training; it does not turn prior efficiency negatives into positives.

See [fixed reference](reference.json), [ordinary reference](ordinary/reference.json),
[summation control](controls/summation-comparison.json),
[attention control](controls/primitive-attention-comparison.json), and
[update qualification registration](controls/fp32-updates-registration.json).
