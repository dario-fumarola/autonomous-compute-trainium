# Elementary difficulty is not explained only by large operands

This is a post-hoc partition of all256 previously scored cases, with no new
model inference or selected endpoint. Every original premise reconstructs to
its frozen expression key. Counts use absolute operand magnitude, including
signed values.

| Operands | Cases | Generated state correct | Ordinary correct | Trained pause correct |
|---|---:|---:|---:|---:|
| Both below10 |77|6|4|3|
| Both below100, remaining cases |120|3|5|1|
| At least one100 or greater |59|4|4|3|

The low overall accuracy therefore also includes weak performance on the
small-operand subset. This does not identify the cause or establish a latent
advantage. Unequal operator distributions and small subgroups preclude treating
the rows as a controlled difficulty curve. The original full-panel findings
and all scientific gates remain unchanged.

Final reaggregation takes0.090CPU wall seconds. Three preceding extraction
attempts stop before any quality breakdown: an omitted document-start BOS
offset, then incorrect assumptions about the calculator's set of distinct
signed literals. Their receipts are retained; exact process durations were not
captured. Two regression tests cover those cases. No model score, source panel,
training computation or active numerical engine was modified.
