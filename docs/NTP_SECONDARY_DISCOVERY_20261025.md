# Secondary NTP candidate, selected on TUNE before MAIN

The completed three-rate NTP recurrent grid has a promising discovery score:
2.917065 at LR .032, versus best tested one-pass NTP 2.938677 and one-pass
future loss 2.935304. All use 301,465,908 real source targets and the same
parameter inventory. Recurrence uses 1.232× estimated training matrix work.
This is one tuning seed and all winners are at the upper grid edge. It is not
an independent confirmation, a compute-efficiency win or a resolved optimum.

This motivates a **secondary discovery readout of plain NTP recurrence**.
The choice is made after seeing these TUNE outcomes and is disclosed as such.
It was registered before MAIN admission or MAIN outcomes. No new training,
source selection, inference exit, forward pass or primary criterion is added.

The original future-objective experiment stays intact. Its primary interaction,
absolute candidate gains and supported-depth gates remain unchanged. Its
positive or negative decision must be reported alongside this secondary result;
the NTP result cannot turn a failed future-objective hypothesis into a success.
Nor would this isolate why the new NTP recipe works: gradient rule, learning-rate
selection and execution differ from the older pilot, so a causal attribution to
full backpropagation alone would need a separate controlled comparison.

After all eight MAIN fits and their numerical gates finish, reuse the existing
fixed final scores and document totals for seeds 5231 and 7283. A plain NTP
candidate advances only if **both** pilot seeds meet all three point gates:

- Recurrent NTP at its declared two-pass exit beats trained one-pass NTP by at
  least 0.01 nats.
- It also beats trained one-pass future loss by at least 0.01 nats.
- Its same-checkpoint supported one-to-two-pass gain is at least 0.01 nats.

Require exact matched source exposure and parameter inventories. The two
inference exits must use identical weights. Use the existing 60/20/20 metric
and all original source groups, with conditional group intervals reported
separately. Neither a favorable same-LR comparison against a weaker baseline
nor an unsupported truncation is sufficient. No incomplete or failed MAIN
cells are silently dropped.

Passing these discovery gates is not terminal success. The candidate still
owes the strongest eligible ordinary/cheap baseline, a separate matched-work
comparison, inference costs, and fresh confirmation with at least five paired
training seeds and new source groups under the existing alpha allocation.
The current consumed DEV readouts spend no terminal alpha and cannot become a
fresh confirmation by changing their label.

The observer reads only already produced MAIN outputs. It requests no chips,
performs no new model forwards or updates, and leaves the original controller,
models, rates, data, schedules and budgets untouched. Its numerical analysis
code and registration are frozen separately. Failures produce explicit stopped
receipts; no parent success or secondary positivity is inferred from status text.
