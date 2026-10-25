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

## Post-hoc split description

The fixed held-out panel is not entirely novel in arithmetic structure:
103/135 expressions have an operator-tree shape represented among the 1,618
TRAIN problems, and 130/135 share an operator multiset. The training set has
506 distinct operator-tree shapes. Numeric values, full expressions and source
groups remain distinct under the original split rules. These counts do not
establish that the required computation was learned or that examples duplicate.

The operation-count distributions are also broadly comparable: 2/3/4/5/6
operations occur in 657/594/149/176/42 TRAIN tasks and 58/44/22/9/2 DEV tasks.
Median no-work prompt length is 26 versus 27 tokens. This weakens a claim that
the entire failure comes from wholly unsupported operations or much longer
held-out prompts. It does not identify a unique cause of the generalization
gap, justify selecting an easier subset or alter the negative result.

This is a model-free descriptive audit performed after the negative pilot,
with zero accelerator use or learning updates. It changes no running task or
evaluation. [Aggregate counts](split-description.json) can be reproduced with
`scripts.describe_natural_chain_split` using the two frozen content hashes.
