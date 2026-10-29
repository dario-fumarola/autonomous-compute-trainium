# Does the code gain distinguish bracket types?

Register before any tied-challenger quality score. Reuse all282 authored
closing-delimiter cases from141 TRAIN-excluded groups, matched long/short by
closer and nesting. This is previously used development data. The earlier
positive raw NLL result motivates this exploratory diagnostic; it does not
make the panel fresh. No cases or original targets are changed.

Freeze vocabulary classes without model scores: decode each public-tokenizer
token, remove leading whitespace and classify by its first character if it is
one of ),] or }. Sum the model's probability within each class, then normalize
over the three classes. Every original target must belong to its recorded
closer class; otherwise stop without dropping cases. There are83/37/98
vocabulary entries and63/36/42 source groups for these classes, respectively.

Primary diagnostic: conditional type NLL on long cases, macro-averaged with
one-third weight per closer type. Also report conditional type accuracy,
short cases, full versus last32 original-token context, interactions and
unmodified original-target NLL. Bootstrap source groups separately within
closer type,2000 draws, seed10033, preserving paired long/short conditions.
Use the existing0.03-nat positive-interval development screen against each
control; neither an interaction nor a punctuation-mass gain can replace a
failed type comparison. Uniform-type NLL ln(3) and accuracy1/3 are descriptive
references. Accuracy is conditional three-class prediction, not free decoding.

The candidate is the fixed mixed LR chosen on broad NLL, executed sequentially.
Controls are the retained selected trained pause and the independently broad-
selected tied ordinary model after all three fits/audits. No structural score
chooses their weights or LR. Report unbracketed control selection explicitly.
The separately registered broad exact-sequential comparison remains primary;
this diagnostic cannot rescue a broad failure or establish general reasoning.

Full context retains the distant opener; last32 removes it in long cases and
retains it in short cases. All source tokens remain original substrings. No
teacher, generated answer or synthetic research example is introduced.

CPU FP64, eight threads,4GiB RSS,60minutes from registration, zero optimizer
updates and chips. Skip unused earlier vocabulary projections in sequential
evaluation only; compare full logits and gradients against the existing exact
reference in software tests, then two32-token actual-weight prefix checks at
1e-8 before scoring. Prefix hidden states and KV entries remain unchanged.
Preserve512GiB host and35GiB disk reserves. Keep row data private, publish
aggregates/hashes. Fresh reserves remain unscored; no new fit is admitted.
