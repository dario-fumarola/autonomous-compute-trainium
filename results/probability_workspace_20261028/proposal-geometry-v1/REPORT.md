# TRAIN-only proposal geometry

A fixed12-window input-only diagnostic compares the original initialization and
both saved workspace checkpoints. No target losses,rewards,held-out scores or
updates. Proposal/input norm ratio grows from0.0211 initially to0.4717 at64.4M
and0.5806 at200.2M targets. First-proposal entropy falls from10.2709 to5.0417
to3.6594nats. These observations show a substantial adaptive signal develops;
they do not show it improves predictions or implements reasoning. The registered
four-arm comparison and all promotion criteria remain unchanged.

17.44seconds CPU,eight threads,zero assigned chips. Source windows are outside
all fitted prefixes but within admitted TRAIN groups. No new corpus selection.
