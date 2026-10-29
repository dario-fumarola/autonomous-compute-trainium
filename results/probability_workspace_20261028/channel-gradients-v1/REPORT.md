# Intermediate channel gradient diagnostic — TRAIN only

At the final checkpoint,the gradient for direct next-token prediction and the
gradient for using the same logits as a one-replay workspace have global cosine
0.09654;37.08% of real positions have negative local dot products. The reader-to-
direct gradient norm ratio rises from0.0080 initially to0.2164 at200M targets.
These describe different local objectives;they do not establish that gradient
conflict causes a training deficit. They are not AdamW update directions.

The fixed slice has12windows from only5documents/groups,with27%web,43%math and
30%code positions. It is not a representative broad-loss estimate. No resampling,
held-out reads,updates or changes to the frozen experiment. An independent CE
analytic derivative agrees within2.33e-10.29.18seconds CPU,zero assigned chips.

All initialization/early/final rows,including the small TRAIN loss comparison,
remain in[report.json](report.json). The registered held-out comparison is still
pending and remains the decision evidence.
