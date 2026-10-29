# Sparse persistent qualification completed

The supplement restores the preserved complete update-103 state exactly in
all 64 ranks, performs the missing generated-slot update104, and verifies all
post-update replicas. An independent Native process reproduces the complete
model, optimizer and RNG state bit for bit. The full 128-window CPU reference
audit passes: overall gradient relative error 0.000217253, maximum family
error 0.000272487 (0.02725%), and loss error 0.000002109 nats. Original limits
remain 2% and 0.003 nats.

Together with the preserved four cold checks and 100 warm measurements, this
completes the numerical/state record. Warm times are 0.298064 seconds for
full ordinary, 0.321776 for chunked ordinary, 0.386800 for generated slots and
0.390778 for constant slots. Generated slots take about 20.2% longer than
chunked ordinary, despite only 0.58% extra nominal matrix work. No hardware
efficiency claim follows from that ledger.

The two previous timeouts remain failed attempts. All three attempts cost
15.304764 assigned chip-hours; opportunity time is separate and overlapping.
The supplemental procedure does not claim to have met the original failed
time budget. Substantial repeated compilation is measured and budgeted.

All weights are disposable and excluded from science. No DEV or fresh reserve
was scored. A separate frozen three-arm learning screen can now be considered;
this qualification is not a prediction-quality result.
