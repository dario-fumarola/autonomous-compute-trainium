# Check the reported budget regime before extrapolating

Rechecked the primary October20 revision of [Truncate or Not?](https://arxiv.org/html/2610.26030v2),appendicesB.3/D.2/D.3.
Its fixed-size measurements favor two fully differentiated loops beyond about30
ordinary-equivalent tokens per non-embedding parameter;the gain is around0.01
nats at50. Its fitted first switch is earlier,around14–17. These are different
empirical and fitted statements,not a universal threshold.

Its smallest body uses2/4/2 layers,2048-position FineWeb-Edu training,untied
input/output tables and about51% recurrent non-embedding parameters. Optimizer
updates add0.06–5.7% to its pass work. Those differences matter before importing
its favorable allocation regime into this repository.

Our deployed small-core model is141.96M parameters,but98.304M are lookup tables,
leaving43.660M elsewhere. The earlier matched-work One's1.238B targets correspond
to about28.35 targets per non-lookup parameter. That arithmetic is a comparison
point,not proof of a crossover. Our2/2/2 core,three value-embedding tables,tied
head,1024 context,natural-source mixture and replicated optimizer costs differ.
The current pilot's short continuation cannot establish convergence or transfer
that paper's scaling frontier.

Do not change the live curriculum protocol or automatically launch a larger NTP
run from this note. After its fixed readouts,a longer full-gradient comparison
would need its own falsifiable budget/exposure hypothesis,strong lean one-pass
controls and measured cost. Preserve both domain and broad-source negatives.
