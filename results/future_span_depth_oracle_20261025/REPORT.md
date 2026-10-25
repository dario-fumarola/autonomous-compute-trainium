# Free document-level depth selection cannot close the current gap

This adaptive retrospective DEV diagnostic was registered after the first
matched-work pair lost and before second-pair scoring. It reuses existing
plain-NTP recurrent predictions at supported depths1/2/3;no new model forwards
or updates occur. Target-informed selection and its cost are free.

| Seed | Best free whole-document NLL | Matched-work one-pass NTP | Oracle gain |
| --- | ---: | ---: | ---: |
| 5231 | 2.586340621 | 2.581289557 | −0.005051064 |
| 7283 | 2.584254559 | 2.581382256 | −0.002872303 |

The free oracle also loses to both future-objective controls. Most documents
select depth3;selection improves only0.000172/0.000190 nats over using depth3
everywhere. Every original document,source group,target and60/20/20 domain
weight is preserved. Endpoint identities and complete cooled work matches are
rechecked;all original aggregate losses reproduce.

Even this optimistic bound cannot recover the current efficiency gap. A
whole-document selector limited to these same recorded predictions cannot
improve on it. This does not bound token-level policies,changed state paths,
new training or architectures. Nominal full-window inference work is reported
separately from actual native latency;no selection or cached-serving cost is
included. No new primary,confirmation claim,TEST access or terminal alpha.
[Exact result and provenance](report.json) · [Registration](registration.json).
