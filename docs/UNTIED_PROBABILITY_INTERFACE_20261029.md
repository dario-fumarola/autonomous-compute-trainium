# Independent prediction and workspace-value tables: qualification first

The budget-cue fit is closed:all supported depths score about3.4231 and its
TRAIN proposals are almost constant. This does not justify another cue,router
or longer version of that fit. The separate fixed-one fit remains much better
(3.24117),but worse than equal-data ordinary(3.21105).

One structural difference from the published probability-workspace method is
still untested:our input/value table also supplies every output logit. AppendixB
of[PonderLM](https://arxiv.org/html/2505.20674v3) explicitly uses untied input and
output embeddings. The pinned Pythia70M configuration likewise unties them.
Other backbone,data,context and exposure differences remain;this experiment
would not be a reproduction of that paper.

For a fixed hidden query h,the tied proposal softmax(hE^T)E has a symmetric
positive-semidefinite Jacobian with respect to h. Independent output rows W
and value rows E permit a general cross-covariance. This is a local structural
observation,not a proof that tying caused the negative result. Previous TRAIN
role gradients did not show cancellation;do not invoke cancellation as evidence.

Implement W as an independent Parameter initialized by copying E,without
consuming another random draw. Every initial prediction at every budget must
match the tied model,while subsequent gradients may differ. The same body,
softmax,residual proposal addition and uniform natural-token objective remain.
No budget cues,gate,router,auxiliary target or new data mixture is introduced.

The table adds16,384,000 parameters:79,184,384 total at the existing actual shape.
Every prospective ordinary control must receive the same untied table and
parameter count. Both input and output tables receive zero weight decay under
the existing table exemption;body matrices retain0.1. New table storage is
charged. Matrix operation counts remain unchanged because the formerly tied
head already executed its dense projection. Runtime/communication/memory may
change and must be measured;matrix matching is not wall-time matching.

First qualify,without scientific fitting:

- CPU initial and mature-weight forward parity,causality,independent replay
  gradients,and exact restoration/optimizer replay. At the copied state,the
  independent input/output gradients must sum to the original tied gradient.
- Separate input-table,output-table and body gradient checks under the existing
  2% criterion;do not hide the new head inside a large body family.
- Disposable64-rank small-array gradient/AdamW/moment and separate-process replay.
- Actual79.18M,batch4/context512 cold checks,100warm updates,grouped four-rank
  full-state capture,exact next-update replay and complete256-window CPU audit.
- Same512GiB host,192GiB worker,35GiB disk and projected-inode reserves. Each
  stage has its own frozen deadline,source hashes and full artifact allowance.

No held-out panel or protected source is read in qualification. Disposable
weights never initialize a scientific fit. No fit is admitted by this document.
If qualification and measured cost are acceptable,a separate registration may
compare fixed-one and ordinary at equal200M natural targets and equal complete
training matrix work,using the original source order and cooldown. That design
must include new ordinary controls,retain the0.01 practical margin,all domains
and all costs,and distinguish the architecture-by-untying interaction from any
absolute advantage. A single unsupported shallow exit is not an iteration curve.
