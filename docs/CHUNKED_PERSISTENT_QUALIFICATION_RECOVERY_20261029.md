# Bounded completion after first compilation timeout

The first attempt passed all four full-shape cold gradient checks and all 100
warm updates, but reached the 20-minute Native phase cap during update-103
replica capture. Sixty of 64 replica receipts and the rank-zero state exist;
that is not complete qualification. No mature audit or restart pass is claimed.
The original attempt and its 5.340969 assigned chip-hours remain recorded.

Measured ordinary/full, ordinary/chunked, generated and constant warm means are
0.297368, 0.323220, 0.388513 and 0.389913 seconds. Generated/full-ordinary ratio
is 1.306506. These are descriptive measurements from the stopped fixture, not a
qualification pass. They satisfy the original throughput limits.

Admit one separately identified recovery from the same disposable random
initialization, with identical data, graphs, optimizer, 104 updates, all cold
checks, full replica capture, fresh-process replay and complete CPU gradient
audit. Existing compiled graphs may be reused. Preserve the original absolute
20:46:23 UTC deadline, 20-minute Native and 15-minute CPU phase limits, every
numerical/throughput threshold and all reserves. No automatic further retry.
No scientific state or learning budget is admitted. Count failed work and
recovery work together when reporting qualification cost.
