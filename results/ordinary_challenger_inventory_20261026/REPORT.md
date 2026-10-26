# Prospective conventional controls: metadata only

A conventional eight-layer, width-640, five-head model fits below the recurrent
candidate's parameter cap: 141,733,120 versus 141,963,840, a 0.1625% shortfall.
It retains 128-dimensional attention heads, a 4x MLP and the existing alternating
value-embedding policy. Eight unique layers add a fourth value table; this is a
width/depth/lookup-allocation control, not an identical-parameter paired model.

| One-pass configuration | Parameters | Nominal operations / global update |
| --- | ---: | ---: |
| Existing family, six layers / width 768 | 141,963,840 | 270,391,480,478,208 |
| Ordinary, six layers / width 768 | 140,781,888 | 266,680,628,734,464 |
| Ordinary, eight layers / width 640 | 141,733,120 | 241,239,591,072,000 |

At the completed full-head fixed2B work budget, prospective control endpoints
are respectively 2.463002B, 2.497546B and 2.760804B real targets, each an exact
integer floor. These are calculations, not trained or scored models. The work
ledger includes attention, vocabulary projection, backward and all 32 replicas'
optimizer matrices; it excludes padding/pointwise/network work and chip time.

The inventory uses meta tensors and original TRAIN count metadata. It allocates
no weights, reads no source examples, makes no model forwards or updates, uses
zero chips and takes 1.377 seconds. The shapes were chosen from the parameter
constraint before their quality was observed. This is a possible stronger
ordinary challenger, not evidence that it wins. Numerical qualification,
independent tuning and a separate frozen budget remain prerequisites.

No running model, queue, architecture, data or deadline changes. No new native
or scientific run is admitted. 288 CPU tests pass. No TEST, reserve or terminal
alpha is accessed.

[Exact inventory](report.json) · [Tests](cpu-tests.json) ·
[Reproducible checker](../../scripts/inventory_ordinary_challengers.py).
