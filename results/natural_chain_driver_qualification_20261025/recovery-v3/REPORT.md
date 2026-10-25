# Ordered-conclusion driver qualified

Both one-pass and supported-depth1/2/4 cases pass all six independent full32-task
CPU FP32 gradient/optimizer audits. Maximum body relative-L2 error0.08803%.
Each case performs121 real qualification updates plus a restored update121.
All32 replicas agree;model/optimizer/RNG restoration is exact. Subsequent-step
state digests differ across restart,so bit-exact future training is not claimed.

The conservative task-only100-warm-update timings are12.880s for One and13.578s
for R. These do not measure an ordinary-NTP-plus-task training mixture. The
initial body is an existing1.005B-target One/NTP checkpoint,with identical fresh
optimizers. No claim of from-scratch task learning or held-out efficacy follows.

The first launch fails on the legacy launcher corpus argument;the second fails
on the native cross_entropy graph break before updates. A CLI-only adapter and
the existing explicit logsumexp/gather CE recover under the original deadline.
CPU value/gradient/fullgraph parity passes. Both failures and all assigned costs
remain preserved. An additional recovery-registration check originally read the
wrong failed log;it rejected before registration/native launch and is recorded
in the private controller log. It was corrected before the third frozen plan.

No scientific continuation adopts these fitted qualification weights. The future
pilot must start again from its common original NTP body and qualify the mixed
objective. No TEST or terminal alpha. All source fragments and states stay private.
