# Measured deployed-path prefill cost

Both paired rounds complete,with the two models swapping physical eight-chip
halves. Weights remain unchanged,the training-only auxiliary head is removed,
no gradients or optimizer updates occur,and every numerical/fallback gate passes.

Median per-rank forward latency,seconds for1024 positions and full logits:

| Model/path | Round1 | Round2 |
| --- | ---: | ---: |
| Trained one-pass NTP | 0.022247802 | 0.022191293 |
| Recurrent NTP,count1 | 0.022510227 | 0.022280628 |
| Recurrent NTP,count2 | 0.025941590 | 0.025937447 |
| Recurrent NTP,count3 | 0.029587257 | 0.029592723 |

The declared count2 path costs1.1660×/1.1688× the trained one-pass median
latency. Within recurrent weights,count2 costs1.1524×/1.1641× count1 latency,
versus1.23038× nominal forward matmuls. These two rounds are timing observations,
not independent training seeds or confidence intervals. Head/input transfer,
networking,tokenization,checking loss and autoregressive KV-cache generation
are excluded from timed forwards. Inputs are preloaded;each call synchronizes.
The prototype is not claimed to be an optimized serving implementation.

Every rank alternates two authentic TRAIN windows. Each path takes100 warm,
100 timed,and four pre/post numerical-check forwards. The measurement totals
52,224 actual native forwards over repeated presentations of64 unique source
windows,and9,741,372,072,394,752 nominal forward matrix operations. Every model
visits both halves. Maximum sampled-logit relative error is0.00045062;all full
masked-NLL checks remain inside0.003 nats. CPU preparation spent103.74 seconds;
native admission through reporting spent171.54 seconds,both inside the common
1800-second cap. Waiting before admission is recorded separately.

This measures fixed-window prefill cost. It does not establish cached serving,
training efficiency,a learned stopping policy or independent confirmation.
[Exact report](report.json) · [Execution costs](native-execution-costs.json).
