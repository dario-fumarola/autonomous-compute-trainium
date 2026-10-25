# Prospective secondary loss stratification

Keep the fixed-two curve's pooled natural-source NLL primary and all its finite
endpoints/controls unchanged. Retain private float64 position losses during the
same independent FP32 score; no extra teacher,training target or research example.
The scalar scoring arithmetic and exact original-target coverage stay identical.

Freeze a separate existing One/NTP work-control checkpoint from seed7283 at
1.238B as the difficulty reference, before the new seed5231 MAIN outcomes. It
is independent of the paired MAIN initialization,avoiding mechanically sorting
by the same baseline whose loss reduction is being measured. Recompute it once
on the same consumed DEV curve cohort with the same independent reference code.
This is diagnostic reuse,not fresh confirmation or an additional fitted baseline.

For the primary2B pair,report gains in ten equal-count reference-loss bins
within each of web,math and code. Ties use fixed original document/position
order. Keep every position and the fixed60/20/20 domain weights; verify that
the bin-weighted gains exactly reconstruct the pooled gain. These bins use
true targets to measure difficulty and cannot serve as a deployable router.
Do not select favorable bins,spend terminal alpha or call token positions
independent replications. No stratified result can rescue a negative primary.

Keep all position arrays and document index records private. Publish only
aggregate counts/means and hashes. There is no exact original UTF8 byte count
in this serialized DEV artifact; do not label decoded-token byte counts as
original-source BPB. The no-synthetic charter also excludes a generated Dyck
research panel. Authentic code probes would require a separate protocol.
