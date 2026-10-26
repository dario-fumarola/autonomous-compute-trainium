# Autonomous computation during pretraining

Can a randomly initialized language model learn to use additional internal
computation to predict unseen natural text better?

Training uses authentic web text, mathematical web text and repository code.
The completed main comparisons use no generated research examples,external
teacher models,added reasoning traces or reinforcement learning. Real-token
negative log likelihood (NLL) remains the broad-text primary outcome. Equal-data and training-work comparisons answer different
questions and are reported separately.

The fixed-two candidate has a small positive development signal,while a useful
advantage over the stronger alternatives remains **unconfirmed**. Completed
comparisons use the same two reused development seeds and consumed DEV panel:

| Model | Training targets | Mean DEV NLL ↓ | Fixed-two advantage |
| --- | ---: | ---: | ---: |
| Fixed two passes | 2.010B | 2.483862 | — |
| Original same-family One,matched work | 2.421B | 2.496426 | 0.012563 |
| Corrected head-free cheap mixing,matched work | 2.446B | 2.494629 | 0.010767 |
| Conventional ordinary8,matched work | 2.761B | Running | Pending |
| Head-free One,matched work | 2.463B | Pending | Pending |

The corrected cheap models trained afresh and passed both full512-window
mature gradient audits. Recurrence wins by0.009852 and0.011682 nats. Their
nominal paired-seed95% interval is**[-0.000858,0.022392]**;one seed point is
below0.01 and both conditional source-group intervals overlap it. This does
not establish a useful gain. The historical failed cheap run remains
inadmissible;the new runs do not retroactively certify its training history.
[Corrected cheap comparison](results/lean_baseline_controls_20261026/main/mixing/REPORT.md).

Against the original same-family One comparator,the mean gain remains
0.012563,with nominal seed interval[0.005601,0.019526],also overlapping0.01.
At1B,the original matched-work gain was only0.004255. Equal-data gains remain
larger. All relevant completed endpoint gradient audits pass.
[Original curve](results/fixed_two_small_core_main_20261025/REPORT.md) ·
[Completed second pair](results/one_control_recovery_20261026/REPORT.md).

Both fixed-two and same-family controls have141.96M deployed parameters.
Ordinary8 has141.73M,slightly below that ceiling,and receives its own three-rate
tuning. It and corrected cheap select0.032 at the upper grid edge;their optima
remain unbracketed. Both ordinary8 seeds are now training concurrently on all16
chips. The bounded queue then runs the declared lean One controls;it stops on
failed audits or unresolved resource limits.
[Qualified tuning](results/lean_baseline_controls_20261026/tune/REPORT.md) ·
[Conventional challenger](results/lean_baseline_controls_20261026/main/ordinary8/REPORT.md).

Matching uses nominal neural/head/attention/backward and replicated optimizer
matrix work. The original candidate's unused training head remains charged;
it is not retrospectively subtracted. Hardware time is reported separately.
The original fixed training path used about2.6% more assigned chip-time than
its original One comparator. Measured prefill is16.4–17.6% slower than that
comparator;costs against the new challengers and KV-cached decoding remain
outstanding. [Prediction cost](results/fixed_two_prefill_20261025/REPORT.md).

No4B extension is admitted before the stronger comparisons. Fresh source
groups,adequately powered five-or-more fresh paired seeds,and terminal alpha
allocation remain required. Five seeds are a minimum,not a power guarantee.
No new TEST,reserve or terminal alpha has been used;the overall goal is
unachieved. [Live status](STATUS.md).

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
