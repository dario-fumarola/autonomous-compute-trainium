# Supported-depth natural-context diagnostic

The syntax-only panel is frozen before model scoring:149 original identifier
uses,with60/60/29 targets at token distances129–256,257–512,513–900. Each band
has at least13 original repository groups. No target text is rewritten. The
selection checks exact tokenizer roundtrip,UTF8 AST spans,unique direct local
binding and dominating assignment; ambiguity and nested-scope uses are excluded.
The data,tokenizer,selection source and private target-position panel have hashes.
All target selection and view/noise refinements precede scoring these context
targets. Separately,the ordinary learning-rate screen produces consumed-TUNE
scores; those are not inputs to this selection or its prospective refinements.

At the low-depth candidate's independently qualified matched-work and equal-data
endpoints,score the same original identifier tokens in four conditions:full
1008-token source prefix at1/2 passes and its last128 tokens at1/2 passes. Both
exits receive training support. Teacher-force only previous original identifier
tokens within a multi-token name. The full sequence,including target-prefix
inputs,fits1024 positions. The fixed1008-token limit reserves space for any
of the prespecified at-most16 identifier tokens,independently of the actual
target's length. Initial noise keys use only the absolute next source index
plus a fixed1000000007 offset,not a document-content hash,target text,length
or selected binding. Both views can construct the key before the prediction.
The original149
targets are unchanged; scoring-view hashes separately record this prospective
refinement before model scoring. The short view retains the exact original token IDs,
absolute positions relative to the full view,and corresponding initial hidden
state slice. Upstream `position_ids` selects the original RoPE cache entries;
do not silently reset the suffix's positions or redraw its hidden state.

Before scientific scoring,qualify the intervention with a tiny random-initialized
independent upstream model. A position-preserving short forward must match the
retained suffix of a full-length forward with block-diagonal causal attention.
Changing the removed prefix must not affect the short or masked-full suffix.
Check both supported exits and the unchanged model. This numerical rehearsal
does not report random-model prediction quality or train on research examples.

Compute G_full=L(full,1)−L(full,2) and interaction=G_full−[L(short,1)−L(short,2)].
Aggregate real identifier-token NLL within each distance band,then weight the
three bands equally. Report every band and both endpoint readouts.
The matched-work endpoint is the primary mechanism readout; equal data is
secondary. Resample
original repository groups jointly across all four conditions,5000 draws with
seed1381,for nominal95% within-model intervals. A positive pilot context-use
finding requires both lower bounds above zero; interaction alone can arise
from harming short-context prediction. This is a descriptive reused-DEV
mechanism readout,not a fresh terminal significance claim or general reasoning.
The main pooled-text0.01-nat useful-effect threshold remains unchanged.

Pinned upstream CPU FP32 model code performs scoring; each endpoint's raw global
gradient must already pass the independent numerical gate. Keep all selected
targets and settings regardless of outcomes. Limit each endpoint readout to
20minutes and the complete numerical rehearsal to5minutes. Raw code,tokens,
positions,per-identifier results and weights stay private; publish hashes and
aggregate bands/intervals/costs only. No teacher,synthetic program,model-written
test,router,RL or new prediction objective is introduced.
