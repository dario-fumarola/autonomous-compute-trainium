# Existing-checkpoint document-depth oracle

On the consumed DEV panel, an optimistic selector that sees the future target
and chooses one depth for an entire document provides little improvement over
the two-pass exit. Even its unconstrained best choices lose to the ordinary
matched-training-work references in all three fitted seeds.

| Family | Constant depth2 NLL | Hindsight minimum NLL | Oracle improvement | Deficit to ordinary matched work |
| --- | ---: | ---: | ---: | ---: |
| fixed | 2.546717216 | 2.546716671 | 0.000000545 | 0.050918656 |
| mixture | 2.588435213 | 2.587784060 | 0.000651153 | 0.091986045 |

These are means over seeds103/211/307 at the old completed1.005B endpoints.
Ordinary's matched-work references trained on1.700B targets. Every cell has the
same150,436,608 parameters and identical603-document /754,750-target DEV panel.
Training-work matching comes from the original canonical ledger; real-token
losses are aggregated within each domain before fixed60/20/20 weighting.
All24 input readout/report pairs are checksum-pinned. Counts, source-group
pairing and the original aggregate scores are reproduced exactly.

The selector minimizes weighted document NLL plus a predeclared multiplier on
dense scoring work. Work includes padded1024-token windows, the full vocabulary
head and every block execution. It pays no cost to decide and may see the true
future, so this is an optimistic retrospective bound. Actual latency is unmeasured.
Mixture depths1/2/3 were trained; fixed1/3 are unsupported exploratory exits.

The bound applies only to whole-document selection among these recorded native
BF16 forwards. It does not bound token/window routing, changed hidden-state
trajectories, newly trained routers, other architectures or true FP32 document
scores. Tiny gains can include numerical selection effects. No selector was
trained or deployed, and no new TEST or terminal alpha was accessed.

The practical decision is to deprioritize a document-level router for this fitted
family. The trained inexpensive-connection control remains the next mechanism
comparison. No current architecture, source, LR or success criterion changed.

The calculation took2.37 seconds including imports, with zero native chips and
zero model updates, while the instance continued its registered training jobs.

[Exact records](summary.json) · [Registration](registration.json) ·
[Cost receipt](execution-costs.json) · [Frontier plot](oracle-frontier.png)
([SVG](oracle-frontier.svg), [PDF](oracle-frontier.pdf)).
