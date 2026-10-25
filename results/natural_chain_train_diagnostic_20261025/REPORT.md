# Post-pilot TRAIN diagnostic

The fixed negative held-out pilot is unchanged. A separately registered hash-
selected sample of256 existing TRAIN problems is scored without updates,using
identical no-work prompts,prefix-derived noise,24-token greedy generation,EOS
and scalar verification. All three whole-panel independent FP32 audits pass.
All native/CPU reports explicitly mark this panel TRAIN,not held-out.

|Frozen checkpoint|Default depth|TRAIN correct|TRAIN answer NLL|
|---|---:|---:|---:|
|recurrent-bridge|2|254/256|0.090956|
|one-bridge|1|255/256|0.083733|
|one-bridge-work|1|256/256|0.005022|

The held-out scores are2/135 for One/bridge,4/135 for R/bridge,and2/135 for
longer One. The models learn the
source training cases and output interface,but this finite answer-only bridge
curriculum does not transfer reliably to new source/case groups. This is
consistent with memorization rather than a general arithmetic procedure;it does
not establish its cause or prove that natural procedural supervision cannot
work. More steps do not repair the declared held-out result.

No optimization,tuning,new source data,TEST or terminal alpha. TRAIN outcomes
are explanatory diagnostics and cannot replace the failed primary endpoint.
