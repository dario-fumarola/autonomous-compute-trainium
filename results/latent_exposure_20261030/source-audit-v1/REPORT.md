# Complete prospective training-prefix expression audit

The CPU-only census checks all 3,386,160 documents and 443,686 source groups
through the proposed ordinary work-matched endpoint. Its 565,715 mathematics
documents contain 18,749 verified calculations and 12,604 unique literal-valued
expression ASTs. There are 1,623 token-alignment rejections, no matched-document
roundtrip failures and no parser-length rejection. No model is scored.

| Prospective endpoint | Real targets | Intact verified calculations | Intact answer tokens |
|---|---:|---:|---:|
| Original screen |128,025,288|617|1,697|
| Equal-data exposure pilot |1,000,060,368|4,398|12,209|
| Ordinary matched-work endpoint |3,782,650,274|16,771|47,126|

These are strict extraction counts, not all arithmetic in natural text. No
verified answer is substituted for a source target or exposed as a thought.

Ninety-four of the original353 development expressions occur in the longer
training prefix. A separately identified subset excludes all of them before
any new fit: **259 cases**, with87/98/74 in the2/3/4–6-operation bins. All three
bins pass the predeclared32-case minimum. Source-group, protected-reserve and
verified-expression overlaps are zero. The original353-case result remains
untouched; this subset is reused development data, not confirmation.

Eight bounded CPU workers complete in171.931 wall-seconds, zero chips, updates
or model scores. Every original token array and source-index hash is retained.
This receipt alone admits no training. [Exact counts](report.json) ·
[Prospective audit protocol](../../../docs/LATENT_EXPOSURE_SOURCE_AUDIT_20261030.md).
