# What the current objective test does and does not reproduce

Primary papers rechecked October25 while the frozen MAIN recurrence fits run.
No current source,objective,rate,endpoint or decision rule changes.

[ConceptLM,sections2–3 and5.2](https://arxiv.org/html/2602.08984v1) feeds predicted
concept representations into its causal token decoder. Its concept module
predicts weighted codebook embeddings;the broadcast representations are shifted
to preserve causality. Ground-truth future concepts are training targets rather
than decoder inputs. Consequently,NTP gradients pass through an inference-time
concept pathway. That is a different intervention from our training-only head,
which is discarded for ordinary prediction. Its architecture-fixed objective
ablation uses30B tokens:WikiText perplexity changes34.30→33.45,while OpenWebText
remains12.64 when the full objective replaces NTP alone. Prediction without the
quantization objective is worse on both. This supports investigation,not an
expectation that our simpler auxiliary loss must work.

[Future Summary Prediction,section2.2 and appendixD.2](https://arxiv.org/html/2510.14751v2)
defines the multi-hot future target with reweighted BCE. Our constant weights,
sparse anchors and linear head are adaptations. The learned-summary method
instead uses a separately trained reverse LM,which is outside this experiment.
Its NLL tables omit the bag-of-words variant;reported downstream gains therefore
do not establish our teacher-free bag-target NLL hypothesis. Even the8B learned-
summary model has slightly worse NLL than NTP on both listed validation sets.
It is inappropriate to transfer a downstream headline into a prediction claim.

Our inference:auxiliary supervision and a predicted representation on the
prediction path answer different mechanism questions. Complete the present
bounded factorial and its original natural-context readout. A later concept-
path experiment would need its own parameter-matched ordinary architecture,
causality checks and complete training/inference-cost controls. This note
registers no new model,loss search,teacher,data source or training run.
