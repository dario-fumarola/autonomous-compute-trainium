# Repeat the frozen numerical-consistency diagnostic at one billion targets

The published one-billion-target latent predictions already use exact sequential
execution. This separate CPU diagnostic asks whether the parallel approximation
used during training remains consistent after the longer exposure. It changes no
model, learning rate, scientific endpoint, primary score or advancement rule.

Reuse the six TRAIN windows, numerical-round list, cached/full-prefix checks and
domain tolerances in the original October 29 convergence protocol. These two
windows per domain begin beyond window 10,000,128, outside the complete frozen
3.783B-target training prefix. They are not a new held-out comparison or an
independent estimate of quality. Score only the mature-audited latent endpoint
from seed 730271, update 15,287. Bind its training plan and audit hashes before
execution. Do not select checkpoints from the consistency result.

Run the unchanged diagnostic implementation: K={0,1,2,3,4,8,16,32}, exact
sequential prediction and state discrepancy, forward KL, and the original
K2/sequential domain tolerances (absolute mean NLL shift <=0.01, KL <=0.01,
K16/K32 KL <=0.0001). K is a numerical solver budget, not a count of internal
thought tokens. A failed tolerance is an engineering finding; it cannot erase
the exact published quality score. A pass does not establish reasoning.

Allow one CPU attempt, 20 minutes, 16 threads and 16 GiB peak RSS, retaining
512 GiB host memory and 35 GiB disk reserves. Zero accelerator allocation,
optimizer updates, new data selection, fresh-reserve scoring or terminal alpha.
The active 16-chip ordinary control remains unchanged. Record concurrent host
work and all diagnostic costs. Any future optimization experiment needs its
own explicit design and measured admission; this readout admits none.
