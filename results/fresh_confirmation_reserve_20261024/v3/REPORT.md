# Fresh natural-text confirmation reserve: data audit complete

The data-only reserve has passed independent qualification. No model has scored
its TUNE, DEV or TEST panels. TEST remains sealed; no terminal attempt or alpha
allocation has been registered.

The repaired scan reuses the exact hashed pre-score pool from stopped executionv2.
It opens one Parquet reader at a time, releases reader/Arrow buffers at each file
boundary, and rebuilds all prior-neighbor coverage from zero. No admission filter,
candidate selection rule, source or overlap threshold was relaxed.

| Role | Web targets | Math targets | Code targets | Total targets |
| --- | ---: | ---: | ---: | ---: |
| TUNE | 250,299 | 250,342 | 256,212 | 756,853 |
| DEV | 250,335 | 250,134 | 250,341 | 750,810 |
| TEST | 250,327 | 251,396 | 251,830 | 753,553 |

All14,700,735 prior accepted IDs were screened. The independent auditor verified
all1,697 selected raw rows and complete token sequences, SQL identity/group
exclusions across prior corpora and panels, mapped-ID equality and complete raw
sketch coverage. Prior exact-ID, prior source-group and cross-role source-group
overlaps are zero. It did not independently rerun every approximate minhash
comparison; the tested inverted screening algorithm and its full prior coverage
remain part of the data gate.

Preparation took2,692.22 seconds, followed by162.99 seconds of independent audit.
Zero native chips were assigned; these CPU intervals overlap the paired native
learning run. The conservative extra memory allowance remained above32.45GiB.
Both earlier stopped attempts and their costs are preserved in their own records.

Data readiness is not scientific confirmation. A future candidate still needs a
newly frozen model/control comparison, resource budget, at least five fresh
paired training seeds, independent numerical audits and the ongoing repeated-test
allocation before this TEST may be opened.

[Independent audit](independent-audit.json) · [Execution costs](execution-costs.json)
· [Frozen-pool provenance](frozen-pool-reference.json) · [Manifest](manifest.json).
