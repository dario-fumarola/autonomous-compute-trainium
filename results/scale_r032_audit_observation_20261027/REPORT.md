# Repeated unchanged-state observation of the0.032 audit rejection

The original rate controller stops after the full512-window mature audit
rejects recurrent0.032. Its original failure artifact contains only a generic
assertion. No held-out quality score or rate acceptance follows.

A separately registered read-only calculation repeats the exact frozen
upstream FP32 audit on the same saved checkpoint,512 packed windows and
captured Native recurrent noise. The only inserted operation serializes
observations before the existing gate. All original source bindings and
arithmetic remain unchanged. Zero Native executions or optimizer updates.

The rejection is reproduced in669.402seconds:

| Check | Observation |
| --- | ---: |
| Overall gradient relative error | 6.066574% |
| Injection/readout family | 7.737491% |
| Transformer body | 6.086506% |
| Value embeddings | 5.336930% |
| Tied input/output | 0.951319% |
| Native minus CPU NLL | +0.00000224972 |
| Independent CPU gradient norm before clipping | 0.102809787 |

Lineage,cursor,targets,clipping metadata and unchanged-model checks all pass.
The2% gradient gate fails despite close scalar-loss agreement. This is an
inadmissible learning-rate history,not a quality comparison or a demonstrated
vendor defect. No threshold is relaxed. The original stopped audit and all
state/gradient/optimizer/RNG evidence remain unchanged. This observation
does not invalidate separately passing lower-rate histories.

Future independent audit wrappers now retain observations before applying
the same gate. The frozen original auditor is not edited. Its source-identity
fixture passes;the full source suite passes435 tests,two skipped.
