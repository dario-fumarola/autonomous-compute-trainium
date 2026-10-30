# Enriched training remains inside the expression-audited source range

An independent metadata check matches all112,057 enriched training documents
against the original indexed source prefix used by the expression-exclusion
audit. Source identity, group, domain, byte-token offset and visible range all
agree. There are zero unmatched documents or range escapes. Domain target
counts sum exactly to the frozen128,029,840-target endpoint.

The check scans3,386,106 original recipe records, reads no evaluation targets,
performs no model inference and changes no source text. It takes9.599CPU wall
seconds and0.681GB peak RSS, zero chips and zero optimizer updates. It overlaps
the active Native fit and does not support a hardware-time comparison.
The prior hard-panel expression exclusions therefore remain applicable to
this enriched prefix; the readout panel and decision rule are unchanged.

[Receipt](report.json) · [Frozen inputs](plan.json)
