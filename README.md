# Autonomous computation during pretraining

Can a randomly initialized language model learn to use additional internal
computation to predict unseen natural text better?

Training uses authentic web text, mathematical web text and repository code.
The completed main comparisons use no generated research examples,external
teacher models,added reasoning traces or reinforcement learning. Real-token
negative log likelihood (NLL) remains the broad-text primary outcome. Equal-data and training-work comparisons answer different
questions and are reported separately.

The fixed-two candidate now has **two positive paired development comparisons**
at matched nominal training matrix work:

| Seed | One pass NLL |Fixed two NLL|Recurrent gain|
| --- | ---: | ---: | ---: |
|5231|2.496663|2.484648|0.012016|
|17293|2.496188|2.483076|0.013111|

Both have141.96M deployed parameters. Fixed-two sees2.010B authentic targets;
One receives2.421B to match work. The mean gain is**0.012563 nats**. All endpoint
gradient audits pass. The nominal two-seed interval**[0.005601,0.019526]** overlaps
the0.01 useful-effect threshold;it does not replace fresh confirmation. At1B,
the original matched-work gain is only0.004255. Equal-data gains remain larger.
[Original curve](results/fixed_two_small_core_main_20261025/REPORT.md) ·
[Completed second pair](results/one_control_recovery_20261026/REPORT.md).

Work includes the common unused training head and replicated optimizer matrices;
it is not a chip-time match. The first retained fixed training path costs about
2.6% more assigned chip-time. Measured full-window prefill is16.4–17.6% slower;
KV-cached decoding and end-to-end serving remain unmeasured.
[Prediction cost](results/fixed_two_prefill_20261025/REPORT.md).

**Stronger controls remain decisive.** The earlier fixed-mixing control fails a
mature gradient gate;its training history cannot be certified by a zero-update
precision repair. Corrected head-free mixing,head-free One and the conventional
eight-layer challenger now pass bounded numerical training/replay qualification.
They still need appropriate new predictive fits;ordinary8 needs independent
tuning. Its141.73M parameters are slightly below the recurrent ceiling.
[Baseline qualification](results/baseline_numerical_recovery_20261026/REPORT.md).

No4B extension is admitted before those controls. The estimated larger-exposure
gain is an untested hypothesis. Fresh source groups,at least five fresh paired
seeds with adequate power,and terminal alpha allocation remain required. No new
TEST,reserve or terminal alpha has been used;the goal is unachieved.
[Live status](STATUS.md).

The earlier sampled-depth small-core study found useful additional computation
at equal data,but no training-efficiency win. These are two paired development
seeds, with independently scored, cooled endpoints:

| Model | Authentic training targets | Mean DEV NLL ↓ |
| --- | ---: | ---: |
| One pass, next-token prediction | 1.005B | 2.614841 |
| Recurrence, next-token prediction, two-pass exit | 1.005B | **2.591095** |
| One pass, approximately matched training matrix work | 1.238B | **2.581336** |

All have **141.96M deployed parameters**. Recurrence improves equal-data NLL by
0.023747 nats, but loses to longer one-pass training by 0.009759. The work
match is within 0.02153%; it includes the common training-only head and
replicated matrix-optimizer work, and does not imply equal chip-hours.
[Completed comparison](results/future_span_work_controls_20261025/REPORT.md) ·
[Original study](results/future_span_main_20261025/primitive-v2/REPORT.md).

The same recurrent checkpoints improve when evaluated with two supported
passes instead of one. This establishes useful dependence on additional
computation; it does not establish reasoning, learned stopping or superiority
over the strongest ordinary architecture. The proposed future-span auxiliary
objective failed its declared interaction test and is closed.
[Mechanism readout](results/ntp_secondary_discovery_20261025/report.json).

The measured two-pass path takes about **17% longer** than one pass in a
synchronized, preloaded 1,024-position prefill benchmark. Cached decoding and
end-to-end serving were not measured. Even a free, target-informed selector
among these saved whole-document depth paths fails to beat the work-matched
one-pass control.
[Prefill costs](results/future_span_prefill_20261025/REPORT.md) ·
[Oracle bound](results/future_span_depth_oracle_20261025/REPORT.md).

The completed cheap-mixing control repeats the input adapter while executing
the core once. At equal data,it scores **2.613655**, versus recurrence's
**2.591095**, across the same two seeds. It does not recover recurrence's
useful gain. At matched nominal work,cheap mixing scores **2.581267** and
also beats recurrence. Extra core execution contributes useful prediction
work,but still does not earn its training cost under these controls.
[Completed control](results/small_core_mixing_main_20261025/recovery-v2/REPORT.md) ·
[Live status](STATUS.md).

The bounded bridge-hiding curriculum is now **closed negative**. With original
human-written arithmetic steps progressively withheld,R solves4/135 held-out
answers versus One's2/135,and its second pass adds no solved answers. Both
curriculum conditions worsen broad-source NLL by about0.011 nats. They solve
over99% of a separate TRAIN sample,showing a severe generalization gap.
[Completed controlled pilot](results/natural_chain_continuation_pilot_20261025/recovery-v2/REPORT.md) ·
[TRAIN diagnosis](results/natural_chain_train_diagnostic_20261025/REPORT.md).

The seven-endpoint follow-up also loses: retaining the already trained
recurrent foundation gives 3/135 correct, versus 4/135 for the full-history
work-matched one-pass control. Broad-source loss worsens in every curriculum
condition. Even with the original human work supplied, these two models solve
only 11/135 and 9/135. The bridge family is closed at this scale.
[Completed history comparison](results/natural_chain_foundation_pilot_20261025/REPORT.md) ·
[Scaffold diagnosis](results/natural_chain_scaffold_readout_20261025/REPORT.md).

The fixed two-pass control is now complete and reported above. Its architecture,
optimizer and source order were frozen before the curve. The new mixing-only
control changes no completed result.
[Fixed-depth protocol](docs/FIXED_TWO_SMALL_CORE_NTP_20261025.md).

The curriculum used source-authored arithmetic rather than generated problems
or solutions. Its extraction,views and verifier remain reproducible;human
procedural supervision is explicitly distinguished from raw-text-only training.
[Original source pool](results/natural_source_chain_pool_20261025/REPORT.md) ·
[Data views](results/natural_chain_views_20261025/REPORT.md).

Earlier studies remain preserved, including the
[original shared/untied TEST comparison](results/frozen_test_20261023/REPORT.md),
[revised fixed/variable-depth TEST comparison](results/variable_depth_20261023/test/REPORT.md),
[longer-exposure compute control](results/mature_fp32_compute_20261024/REPORT.md),
[cheap-connection comparison](results/depth_connection_20261024/REPORT.md),
[small-core pilot](results/parcae_low_depth_pilot_20261024/REPORT.md) and
[fixed-pause negative](results/learning_curve_20261022/REPORT.md).
The previous [1.7B repository](https://github.com/dario-fumarola/useful-thoughts-trainium)
and [archived pause pilot](https://github.com/dario-fumarola/pause-pretraining-trainium)
remain intact. No completed recipe has independently confirmed a minimum useful
training-compute advantage against the required strongest eligible controls.

A terminal positive claim requires a frozen candidate, fresh source groups,
at least five fresh paired training seeds, the declared useful-effect threshold
and correction for repeated confirmation attempts. No such confirmation has
been opened for the latest candidate.
[Ongoing charter](docs/ONGOING_RESEARCH_CHARTER.md) ·
[Data rules](docs/DATA.md) · [Trainium notes](docs/OSS_NOTES.md).

Install CPU test and analysis dependencies with
`python -m pip install -e '.[test,analysis]'`, then run
`python -m pytest -q`. Some reference-parity tests additionally require the
pinned upstream checkout specified in their preparation documents. Numerical
software fixtures are not research examples. Source contents, weights, full
optimizer states and native environments stay outside this public repository;
CPU tests do not independently reproduce the scientific training runs.
