# Matched memory explains a small part of the prediction gain

The new control completes all2048 fresh-initialization updates and passes its mature numerical/recovery audit. It learns with same-domain,source-group-disjoint earlier memory while preserving each original query/current slot. Every one of the8192 donor views is used exactly once. No base parameters or vocabulary weights change.

| Training memory | Authentic evaluation memory | TRAIN donor1 | TRAIN donor2 |
| --- | ---: | ---: | ---: |
| Matched to each source | 2.275685 | 2.276711 | 2.276294 |
| Independent source | 2.276662 | 2.276587 | 2.276671 |

Raw C0 remains2.282112;the parameter-matched local correction scores2.277691. Under authentic evaluation memory,the independent-memory-trained control retains84.8% of the matched-trained branch's gain. Matched training contributes0.000977nats,conditional95%[0.000338,0.001553]. This is a small development effect,below the registered0.005 useful-content threshold;it is not evidence of equivalence or a training-seed estimate.

The matched-trained branch benefits from authentic rather than donor1 memory by0.001026,[0.000285,0.001875];the donor2 contrast is0.000609,[−0.000041,0.001295]. The corresponding training-by-evaluation-memory interactions are0.001101,[0.000311,0.001948],and0.000600,[−0.000116,0.001290]. The two-map content gate fails. Independent-memory training produces nearly identical scores under its three evaluation conditions.

Every aggregate/domain cell is retained in[report.json](report.json). Both trained branches still worsen math relative to frozen C0. Original C0 features already contain prefix context,so this isolates additional source-matched memory use by the branch;it does not remove all contextual information or prove absence of reasoning in the backbone. Fixed donor maps and reused development groups limit the conditional intervals. No fresh reserve was scored.

The result strengthens a narrow diagnosis:most of this branch's useful correction does not require source-matched earlier memory during learning or evaluation. It does not establish why the architecture supplies that correction. Sharp attention alone remains insufficient evidence of useful rereading,and no iterative or efficient-thinking claim passes.

Ten focused CPU tests pass. Prefit and mature full-size efficient-FP32 versus independent-FP64 checks pass on web,math and code;maximum mature named-gradient relative error3.383e-6. Full model/AdamW/query-RNG/counter restoration and a disposable next-update replay are exact. Scientific checkpoints remain unchanged by the audit,and authentic saved scores reproduce within the registered1e-5 tolerance.

The fresh fit takes329.62CPU wall-seconds;the complete controller410.36seconds,including the3.14-second mature audit and evaluation. No new Native work or backbone update occurs. The measured training matrix work is2.18083541123072e14 operations,including the frozen vocabulary projection's backward path and extra donor projections. Shared feature/C0/comparator costs and reserved-instance opportunity are retained separately in[the ledger](complete-costs.json). This is equal branch updates/parameters,not equal total training work or a deployable donor-memory method.

The current static-memory branch is not promoted to a larger run or confirmation. A later test of internally updated memory must be a separate registered mechanism,with supported exits and the independently trained one-step control retained. [Frozen protocol](../../docs/RESIDUAL_INDEPENDENT_MEMORY_TRAINING_20261028.md).
