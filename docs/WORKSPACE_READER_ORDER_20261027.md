# One prefix-preserving reader-order check

The inline reader predicts from prefix/open/work/close. Even empty framing
changes the immediate continuation context and costs1.308nats after bootstrap.
Test exactly one alternative order,open/work/close/prefix,on the same calibrated
weights and ten previously consumed TRAIN diagnostic anchors. Generation is
unchanged and prefix-only;reader sees the same observed information in another
order. No true continuation is available to workspace generation or selection.

The natural prefix is uninterrupted at the end of this reader input. Both
orders have the same token count and use uncached complete recomputation.
The alternative requires another prefill and cannot reuse writer-prefix K/V;
charge that actual cost. No new normalization,heads,optimizer or reader fit.
This is one declared interface candidate,not a prompt/length search.

Before policy optimization,diagnostic gate A is empty-framed minus raw mean
NLL at most0.05. Gate B is privileged true-future-copy gain at least0.1nats on
anchors with ORIGINAL frozen raw NLL at least0.1. The latter excludes an
already near-perfect target with no useful improvement room. It is a labeled
software positive control,never a generated thought or performance result.
These small-sample point gates qualify an interface;they do not establish
generalization,noninferiority or efficient thought. Preserve a failure.

Report raw,empty,own,independent donor,fixed-pause and privileged-copy scores
for both orders. Use G4,length8,horizon4,temperature1,original per-anchor seeds,
no optimization and unchanged weights. No further interface variant or policy
fit is admitted automatically. CPU16threads and15-minute cap.
