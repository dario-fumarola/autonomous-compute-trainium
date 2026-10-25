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

## Batch and update-count qualification

Rechecking appendix F adds another regime difference. Around its 1B-token
setting, the paper uses 64 sequences of 2,048 positions per update. Our earlier
small-core comparison uses 512 sequences of 1,024 positions: four times as
many positions and roughly one quarter as many updates at the same exposure.
That is a concrete optimization variable, not proof that batch size caused
our deficit. Its learning-rate/batch rule cannot be copied directly into our
different Muon recipe. A smaller batch also repeats our replicated optimizer
matmuls more often; charge that work and retune both arms if it is investigated.
No change to the live curriculum, no new run admission or guaranteed crossover.

Similarly, the older 150M connection control consumes about 1.098e18 declared
neural training matmuls at 1B targets, whereas the newer small-core one-pass
study uses about 0.528e18 including its declared optimizer accounting. Their
same-data NLLs alone cannot rank their compute efficiency. A future candidate
must face strong controls at the resource budget it actually spends.
