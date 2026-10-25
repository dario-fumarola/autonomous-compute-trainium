# Calibrating the supplied long-budget review

The useful recommendation is to measure the ordinary learning curve and the
return on additional depth under strong matched controls. That supports the
already frozen1B/2B curve. It does not establish a30B crossover,justify a
seven-architecture ladder or change the active experiment's outcomes/threshold.

## What the primary sources actually constrain

The claim that no dense looped model wins at matched training compute is too
broad. [RINS,section3](https://arxiv.org/html/2502.07503v3) explicitly fixes model
size and training compute,giving recursive models fewer tokens. It reports
language-model wins after an initial ordinary advantage. Its early-half topology
and roughly200B/500B-equivalent regimes differ from ours. This is evidence that
a fixed-size win is possible,not a prediction that our model will cross.

The [116-run iso-depth study](https://arxiv.org/html/2604.21106v3) asks a different
allocation question and finds a worse optimized compute frontier for its
baseline loop recipe. Its0.459 is an exponent in `r**phi`,not a constant46%
replacement value for each additional pass. At two loops,the fitted recurrent
contribution is about1.3746 times its unique size. That architecture-specific
fit cannot be combined with another corpus's Chinchilla coefficients to diagnose
a precise0.09-nat missing optimization gain or assign a fraction of causal blame.

The recent [full-BPTT study,appendixD.3](https://arxiv.org/html/2610.26030v2)
reports fixed-size two-loop gains beyond roughly30 ordinary-equivalent tokens
per non-embedding parameter,about0.01 nats at50. When model size is free,the
conclusion changes. Its architecture,data,normalization and small optimizer
work differ from ours. These are empirical recipe-specific observations,not
universal minimum-token rules. They support a bounded fixed-two curve without
requiring us to accept either inevitable failure at1B or guaranteed success30B.

## Apply our actual inventory and measurements

The retired20-block recipe's AdamW/LR restriction is not the current recipe.
The current model already uses contractive learned injection,full gradients,
primitive FP32 attention and the qualified Muon/AdamW implementation. The two
MAIN arms independently selected0.032 within the same bounded grid. Absolute
Muon and old AdamW learning rates are not directly comparable.

There are141.96384M deployed parameters,of which98.304M are lookup tables.
The43.65984M elsewhere receive about23.02 targets per parameter at our1B point
and46.03 at2B. Calling the current experiment simply7TPP hides that inventory;
calling our non-lookup count identical to another paper's non-embedding count
would also be unjustified. The fixed-two work overhead is about20.5%,not the
retired large-core recipe's69%.

The qualified eight-chip One measurement is476,220 real targets/s. A naive
30B extrapolation is about17.5hours for that job before additional overhead.
Two simultaneous eight-chip jobs increase campaign throughput;they do not
automatically halve the time for one fit. The review's4.8-hour number is not
a measured budget for our current geometry. Longer exposure or corpus extension
needs a new measured budget and explicit treatment of unique versus repeated
data. The active curve stays within the frozen natural stream without wrapping.

## Statistics and diagnostics

Five paired seeds do not automatically confirm a useful0.01-nat effect. The
charter requires the lower interval bound to exceed0.01,not merely zero. At the
first allocated two-sided alpha0.025,df4 gives a critical value3.4954. Illustrative
paired SDs0.003/0.005 require observed means above about0.01469/0.01782 to clear
that bound. These are algebraic examples,not an estimated power calculation or
measured variance for the pending2B study.

Adopt independent-reference loss deciles only as a secondary diagnostic. Keep
pooled NLL primary. The serialized DEV has token arrays and provenance but no
exact original UTF8-byte count,so do not invent original-source BPB from decoded
tokens. A generated Dyck evaluation would violate the current no-synthetic rule.
The final bridge scaffold test also cannot distinguish initial failure from
later forgetting;that limitation is recorded without reopening its closed recipe.

No fresh confirmation,terminal alpha,router,new objective or30B launch follows
from this literature check. Finish the registered finite comparison first.
