# Fresh source diversity improves transfer;iteration remains unproven

All six fixed endpoints complete and pass the mature CPU gradient/full-state audits. The source-diversity gate passes;the original iterative mechanism gate fails. This is one reused development seed and116 evaluation source groups.

| Branch | Fresh8192-view NLL | Repeated2048-view NLL | Fresh advantage |
| --- | ---: | ---: | ---: |
| linear | 2.283227 | 2.297520 | 0.014293 |
| one | 2.275685 | 2.289734 | 0.014049 |
| two | 2.276055 | 2.286747 | 0.010691 |

Frozen C0 remains2.2821122052683624 in every condition,within9.45e-7 of its historical value. Fresh one-step improves0.006427nats (conditional95%[0.002270,0.010113]);fresh two-step improves0.006057 ([0.001879,0.009755]). Two steps trail one by0.000370;the paired interval spans zero. Two steps beat the linear control by0.007172,but do not meet either the0.01 absolute gate or the0.005 one-step gate.

At identical branch update counts,two-step fresh training beats repetition by0.010691nats ([0.006763,0.014301]). The corresponding fresh-data advantages for linear and one-step are0.014293 and0.014049. All repeated conditions worsen held-out prediction. This supports source diversity as a contributor to the earlier transfer failure,not an efficient iteration claim.

The post-fit diagnostic evaluates the original first512 TRAIN views. Two-step utility on originally trained labels is0.022574 after fresh training and0.069975 after repetition;held-out utilities are+0.006057 and−0.004634. Repetition improves this training subset while hurting held-out performance. The fresh diagnostic does not represent all8192 trained views. No saved weights are modified.

Each arm ends at2048 updates with262144 uniformly sampled label exposures. Matching512-update prefixes are restored exactly;the new1536 updates preserve optimizer/RNG state. Fresh training uses4315 source groups,disjoint from evaluation groups. No source texts,tokens,features or weights are public.

All named FP32 branch gradients agree with FP64;largest relative error 6.98786518e-06. Model/AdamW/RNG restore exactly and checkpoints stay unchanged. All16-chip feature extraction passes three independent HF references with no fallback,48.6505whole-phase seconds/0.216224assigned chip-hours. CPU fits take338.9–406.7seconds concurrently;final CPU audit/diagnostic252.42seconds. Prior features and512-update prefixes remain in the [cost ledger](complete-costs.json).

This is equal branch updates,not equal total training cost:fresh data requires extra frozen features. It does not beat a work-matched ordinary continuation,establish reasoning,learned stopping,test-time scaling or random-init pretraining. No significance/equivalence claim across training seeds is available. The two-step branch is not promoted. The separately registered parameter-matched local nonlinear control subsequently scores2.277691;the one-step attention advantage is0.002005nats. See [the complete attribution test](../residual_local_control_20261028/REPORT.md).

[Frozen design](../../docs/RESIDUAL_DATA_VOLUME_20261028.md) · [All scores and intervals](report.json) · [Mature audit/TRAIN diagnostic](train-diagnostic.json)

Descriptive [domain means](domain-means.json) show fresh one-step utility+0.007711web,−0.005344math,+0.014347code;fresh two-step has the same signs. This is not evidence of mathematical reasoning.
