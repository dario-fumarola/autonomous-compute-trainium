# Frozen TEST confirmation after this campaign's models are final

Keep TEST sealed while the untied control and longer pair are training. Once
all specified endpoint weights and analysis rules are frozen, confirm only
these trained paths: the three original ordinary/shared pairs at 1B, the single
untied seed 23 endpoint at 1B, and the ordinary/shared seed 23 pair at 3B. Include
every completed specified endpoint regardless of its DEV score; report missing
models explicitly. No exploratory unroll or model selection uses TEST.

First calibrate the generic trained-path sharded evaluator on DEV against
original scalar scoring for ordinary, shared and untied architecture templates.
Require absolute weighted-NLL agreement within 0.0001, plus CPU-forward agreement,
full real-target coverage, unchanged weights and replica equality. The untied
loader retains its extra independently trained blocks. Only its trained depth
is permitted. Use an immutable release manifest containing exact full-checkpoint
SHA256s, allowed trained depths, TEST source SHA and evaluator source SHA.

The original three paired initializations are the sole replicated confirmation.
Use the same 60/20/20 weighted real-target NLL and 0.01-nat rule, with paired-seed
95% Student-t interval (df = 2). The untied and 3B single pairs are diagnostic;
source-group bootstrap intervals condition on those weights and do not estimate
seed variance. Publish every declared contrast and all domains. Do not replace
the original DEV result or imply these single pairs are replications.

Freeze analysis source before release. Once TEST is opened, finish reporting
this fixed campaign without further model/architecture/LR choices on it.
Later scientific variants require a separate held-out panel. Pure software
performance qualification does not select models using these scores.
