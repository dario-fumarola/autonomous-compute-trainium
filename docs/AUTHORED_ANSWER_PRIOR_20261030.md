# Descriptive TRAIN-only answer-prior calibration

Before interpreting the small exact-answer counts as calculation competence,
measure a predictor that ignores the evaluation prefix entirely. From the
existing verified calculation pool, count literal answer token sequences whose
full premise and answer were visible within one window of the original1B
TRAIN prefix. Select the most frequent complete answer sequence; break ties by
token-ID tuple order. Freeze this TRAIN-only choice before reading DEV answers.

Use separate elementary and multi-operation TRAIN counts for their respective
existing panels. Report complete token-sequence equality and answer-prefix
equality, with every original operator/operation bin. The latter matches the
model readout's answer-prefix convention; neither is an NLL comparison or a
learned reasoning mechanism. Source-dependent formatting can make a constant
token guess weaker than a numeric-value prior, which is not substituted here.

This is a post-hoc development calibration, not a newly untouched benchmark or
a change to any primary. Read no model weights or fresh reserve. Preserve the
original published scores. Limits:180 CPU wall seconds,8GiB RSS,16MiB artifacts,
zero chips/updates/model inferences, and existing host/disk/inode floors.
