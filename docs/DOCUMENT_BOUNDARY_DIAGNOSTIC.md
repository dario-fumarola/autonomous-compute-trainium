# Read-only document-boundary diagnostic

On seed23 terminal ordinary and shared-recurrent checkpoints, score32 original
packed TRAIN windows in two attention modes: original causal attention and
causal attention restricted to the current document. Keep every input token,
position, target and loss mask identical. This changes only attention across
inserted document boundaries; it does not create or rewrite source text.

Freeze selection before scoring: draw256 distinct indices uniformly from the
seen stable900M prefix using NumPy default_rng(20261023), sort them, and retain
the first32 with an inserted BOS after position0. Use context1024 and the
original mapped source arrays. Both models have seen every selected target.

Report token-weighted NLL by domain and the fixed60/20/20 aggregate in both
modes, plus the architecture interaction. This is an exploratory single-pair
format-sensitivity readout on TRAIN, not a DEV/TEST result, population
performance estimate, training intervention or proof of the cause of the
original deficit. No optimizer update or outcome-based model selection occurs.
