# Remove reader-side framing,retain exact zero-work identity

The first reordered reader preserved the original natural suffix but still
inserted prose before it. Its point empty-frame penalty0.0556868 misses the
registered0.05 gate;keep that failure. The privileged-copy gate passes.

Test one structural refinement: the reader input is simply workspace+prefix,
with no reader-side opening or closing prose. Zero workspace yields exactly
the original prefix,without a bypass or output mixture. The writer still
generates from prefix+opening without seeing future text. This changes the
reader interface only;charge its second prefill and all processing. It does
not make sampled thoughts informative or save their generation cost.

Use the same calibrated weights,ten consumed TRAIN groups,seeds,G4,length8
and horizon4. Freeze the unchanged point gates: empty-minus-raw at most0.05,
and privileged future-copy mean improvement at least0.1 on the original
raw-NLL>=0.1 subset. The latter is a diagnostic positive control,never a
sampled thought or inference input. Report own,donor and fixed-pause controls
even if they lose. No optimization,TEST or new confirmation data.

Passing permits further policy qualification,not an efficiency or thinking
claim. No further framing or length search follows automatically. CPU16
threads,15-minute operational cap;preserve all outcomes.
