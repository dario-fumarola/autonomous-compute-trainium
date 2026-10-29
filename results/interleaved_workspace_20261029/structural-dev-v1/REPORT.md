# Exact sequential prediction on authored code dependencies

The preregistered readout completes on all282 existing authored closing-delimiter
cases from141 training-excluded source groups. It uses the already selected
original fixedK2 checkpoint, executed with exact sequential latent generation,
and the independently selected ordinary and trained constant-pause controls.
All three received128,025,288 authentic training targets. No weights changed.

| Predictor | Long-dependency NLL | Matched short-dependency NLL |
|---|---:|---:|
| Sequential generated latent |1.840254|1.945737|
| Ordinary |2.189789|1.990619|
| Trained constant pauses |2.044123|2.043581|

The latent model's absolute long-target gain is **0.349535nats over ordinary**
(conditional95% source-group interval **[0.155170,0.565424]**) and
**0.203869 over pauses** (**[0.038329,0.372326]**). Both registered absolute
gates pass: gain at least0.03 and lower interval bound above zero. These are
development diagnostics, not multiplicity-adjusted confirmation intervals.

The long-minus-short advantage is0.304653 over ordinary [0.062227,0.559614],
but0.106025 over pauses [-0.108850,0.328446]. Thus the evidence for a specifically
long-dependency advantage over an equal nominal inference-work control is
weaker than the absolute long-target improvement.

Removing all but the last32 original context tokens worsens latent long-target
NLL by0.712995 [0.508133,0.927183]. Ordinary and pause models also benefit from
full context. The latent model's *additional* full-context benefit over ordinary
is0.058100 [-0.133127,0.258550], and over pauses0.160571 [-0.030253,0.353835].
Both intervals include zero. Context dependence alone does not identify the
internal algorithm or prove superior use of distant information.

This is a concrete structural prediction lead on original natural code,
beyond the broad-text equal-data lead. It remains one training seed, with a
previously used development panel and independently selected learning rates.
The latent model costs4.937times ordinary nominal training matrix work. It does
not establish broad compute efficiency, general transferable reasoning,
autonomous allocation, or a supported multi-thought scaling curve. Fresh
reserves remain untouched and this receipt admits no larger fit.

The two independent FP64 cached/full-prefix checks agree within1.78e-14nats.
The CPU-only readout takes1590.21seconds, with peak1.304GB RSS, zero optimizer
updates and zero accelerator chips. All contrasts, including inconclusive ones,
are retained in [report.json](report.json).

[Registered protocol](../../../docs/INTERLEAVED_STRUCTURAL_READOUT_20261029.md).
