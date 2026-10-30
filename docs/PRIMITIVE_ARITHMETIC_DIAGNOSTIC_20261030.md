# Is elementary arithmetic competence itself missing?

The frozen multi-operation primary remains unchanged. Exact answer-prefix
accuracy there is near 2%, and most answer-NLL gain comes after the first
teacher-forced answer token. A separate diagnostic asks whether the three
completed billion-target models perform reliably even on single operations.
It cannot retrospectively replace their failed or favorable primary results.

Select only original source-authored integer equalities with exactly one add,
subtract, multiply or divide operation; integer operands have magnitude at most
1,000 and integer results at most 100,000. The bounded verifier checks the
authored answer; it never writes a replacement label. Exclude copied answers,
symbolic tails, unsupported expressions and already-visible answer strings.
Keep the complete original premise and up to 256 original context tokens,
without prompts, worked steps, reformatted expressions or generated cases.

Scan at most the existing 100,000-document limit in the mathematics hash bucket
excluded from every indexed training fit. Exclude both protected reserves and
all other protected source groups using the existing registry. Use a fixed hash
order, one source group and literal-valued expression per panel, up to 64 cases
per operator. Require at least 32 per operator; otherwise close with insufficient
coverage. Freeze source hashes and every case before reading any model score.

Unlike the harder expression-transfer panel, elementary facts are **not** removed
merely because their exact expression may have appeared in training. Report this
explicitly: source groups are distinct, but expression overlap is unmeasured.
The question is basic performance, including potentially familiar facts, not
generalization to novel expressions. A positive result would not establish
compositional transfer or reasoning.

If coverage suffices, a separately source-bound CPU readout may compare all three
fixed equal-data endpoints: seed 730271, update 15,287, generated persistent
states, strongest tested ordinary and trained pauses. Use the existing exact
FP64 sequential/batched scorer and numerical checks. Report full-answer NLL,
first-answer NLL, exact answer-prefix accuracy and every operator, with equal
weight across operators and conditional source intervals. No model, prefix or
temperature selection; no useful-effect gate or fit follows from this diagnostic.

Preparation: one CPU attempt, 20 minutes, 8 GiB RSS. Readout, if separately
registered: at most 20 minutes, three 8-thread workers, 6 GiB each. Retain
512 GiB host, 35 GiB disk and inode reserves. The main 16-chip ordinary control
continues unchanged; all concurrent host work is charged and disclosed. No
new training, fresh-reserve scoring, terminal alpha or efficiency claim.
