# Complete four-cell training qualification

All four paths pass the fixed numerical,optimizer and recovery gates. Each
performs121 warm-run global optimizer updates plus one fresh-process replay
update:488 global qualification updates total,with32 replicas per job.
These are qualification weights;zero scientific fitting or held-out selection.

| Cell | Real targets/s per8-chip job,100 warm updates | Trained body gradient error | Trained auxiliary-head gradient error |
| --- | ---: | ---: | ---: |
| one-ntp | 476,220 | 0.0914% | 0.0000% |
| one-future | 476,214 | 0.0615% | 0.0091% |
| recurrent-ntp | 325,926 | 0.1127% | 0.0000% |
| recurrent-future | 325,879 | 0.0512% | 0.0086% |

All12 independent initial/trained/replay audits pass2% body/family/head gradient
and optimizer-update/moment gates,and0.003-nat objective gates. Complete512
original training windows and independently constructed dense future targets
are checked. Replay gold is reused only after exact state/source/depth/noise
pairing. All32 replica model/optimizer states agree,and full saved model,
optimizer and per-rank CPU RNG restore exactly. This does not claim future
bit-identical native trajectories or a scientific prediction improvement.

Assigned native time is3.885 chip-hours;the full qualification window is34.62 minutes,or9.233 whole-instance opportunity chip-hours,including CPU-only audit tails. These ledgers are separate quantities.

The separate scientific-driver integration check follows. Scientific TUNE/main
requires its complete independent pass and the prospective measured-budget
screen. Fresh initialization seeds4127/5231/7283 remain untrained here.
Confirmation sources,seeds and alpha remain untouched.
