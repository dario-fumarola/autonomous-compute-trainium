# Prediction cost after the complete fixed-two curve

Measure the frozen primary pair:fixed2 at2.0098B original targets and its
One2.4206B work control. Wait without owning chips until all six scientific
endpoints and independent audits finish successfully. Timing runs regardless
of their quality ranking. A stopped predecessor prevents admission.

Use actual trained prediction weights with the common training-only head
removed. Freeze64 original TRAIN windows beginning at window2,621,440,
context1024,the original exogenous state-noise policy and local batch1.
Prepare independent pinned upstream CPU FP32 references on the same complete
target masks and eight full-vocabulary logit positions per window. No fresh
held-out source groups or TEST are used for this implementation measurement.

Only the trained operating points are timed:One depth1 and fixed depth2.
The native benchmark derives from the qualified previous prefill benchmark;
it replaces only the hard-coded1920-update endpoint condition with each exact
registered completed cooldown. Retain0.003-nat loss and2% logit relative-L2
agreement gates,unchanged weights,no gradients and no optimizer updates.

In each native round,use two isolated eight-chip/32-rank jobs with authentic
inputs preloaded. Per rank,perform100 warm forwards,100 synchronized timed
forwards,and four numerical-check forwards. In round1 One gets chips0–7 and
fixed2 gets8–15;in round2 fixed2 gets0–7 and One gets8–15. Wait for the actual
first lease before starting its sibling;check reported physical assignment.
Report each round's latency ratio,absolute medians/p95,matrix work and all
CPU/native/setup costs. No throughput gain is assumed from nominal work.

The entire post-MAIN stage is capped at1800 seconds,including at most300
seconds for independent CPU references. Each native phase is capped at660
seconds;four phases run as two paired rounds. The previous same-model-shape
measurement took about50–67 seconds per native phase including checks,with
median forward latency22–30 milliseconds. This cap retains substantial setup
and failure allowance. The deadline is not reset between CPU and native work.
Preserve512GiB host MemAvailable,35GiB filesystem free and10,000 free inodes.
Cancel only owned workers on ordinary failure;retain stopped work and receipts.

This measures full-logits,preloaded fixed-window prefill at batch1 per core.
It does not measure KV-cached token decoding,end-to-end serving or a saturated
batch. Do not present prefill throughput as decode throughput. The scientific
quality endpoint,threshold,confirmation reserve and alpha allocation remain
unchanged;this read-only measurement cannot establish the research goal alone.
