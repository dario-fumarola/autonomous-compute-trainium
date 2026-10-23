# Existing-checkpoint transfer screen — 2026-10-23

This secondary study asks whether the completed matched post-document trace
checkpoint contains useful features that its immediate source NLL missed.
It is a bounded frozen-backbone readout experiment, not full fine-tuning,
exact continuation, or a claim about autonomous computation. It runs on CPU
alongside the new recurrent training without occupying accelerator chips.

Compare all four original matched raw/trace checkpoints (source orders 11/12).
Keep untouched Qwen3-1.7B-Base and both mixed-source checkpoints as practical
references; these are not source-matched causal controls. Pin original model
file SHA256s from the retained registry and verify exact state loading. Preserve
all original files. The two source orders share a pretrained foundation and
are not independent foundation-model replications.

Use human-authored [ARC science exam questions](https://huggingface.co/datasets/allenai/ai2_arc)
at revision 210d026faf9955653af8916fad021475a3f00453, and
[RACE reading examinations](https://huggingface.co/datasets/EleutherAI/race/tree/e30efe648089df42c548e93faa9c5f1816e2c44f).
No generated questions, reasoning traces, teacher outputs or synthetic distractors
are added. Use official TRAIN and validation only; no official TEST access.
The prospective data manifest retains 1,116/295 ARC TRAIN/validation questions
and 256/128 RACE TRAIN/validation passages with one question each. Include only
four-option prompts fitting 512 tokens without truncation. Select RACE passages
and questions by fixed hashes before model predictions. Exclude exact shared
questions/passages across actual adaptation/evaluation sources. This does not
prove absence from the mature base model's historical pretraining corpus.

Freeze each model's backbone. Extract its final normalized hidden state at
the last prompt position and original vocabulary logits for the four answer
labels A/B/C/D. Qualify padded batched extraction against individual prompts
before processing the dataset. Store features privately, never in Git.

Adapt a four-class linear residual readout on those hidden states, added to
the original four label logits. Normalize the hidden vector to unit RMS and
divide by sqrt(width). Initialize the residual weights/bias to zero, so the
unadapted predictor is exactly the original label predictor. Use AdamW,
batch 64, 512 updates, weight decay 0.01, three paired order seeds
101/202/303. Fit tasks separately and report learning curves at 0/64/256/512.
Choose one common LR from 0.003/0.01/0.03 on the untouched BASE's TRAIN-only
80/20 group split, averaging the two task NLLs after the fixed 512 updates.
Then freeze that LR and use all TRAIN examples for every model. No checkpoint
receives its own tuning or early stopping; no validation outcomes select the
common recipe. A model changing the final features is exactly the intervention
this readout can detect; benefits requiring backbone adaptation remain open.

Primary readout: mean validation NLL with equal task weights, after 512 updates,
averaged over the three adaptation order seeds. Also report accuracy, both
task metrics and the unadapted predictor. Compare trace versus its paired raw
checkpoint in each original source order. Bootstrap paired source groups,
carrying all repeated adaptation-seed scores together (2,000 draws, seed 17).
These intervals condition on the fitted original checkpoints; three readout
seeds are not three new pretrained backbones. A useful readout gain of 0.01
nats requires positive lower bounds above that threshold in both source
orders. Upper bounds below the threshold rule out that size of benefit for
this particular readout; otherwise report inconclusive. Report all outcomes.

Bound feature extraction to two CPU hours after a TRAIN-only speed rehearsal.
Record actual model calls, original prompt tokens, wall time and CPU threads.
Do not expand the probe or alter the frozen primary recurrence design based on
these scores. A weak or noisy readout does not establish that all downstream
transfer is absent.
