# Joint-exit optimization screen: qualification first

The 79.18M untied probability workspace loses to both ordinary controls after
200M real targets. It also has an early training-loss lag. Untying increased
proposal variation without improving absolute prediction; the structural
readout supplies no efficiency win. Those results remain closed negatives.

Test one optimization hypothesis: directly supervise the already-computed
first and final predictions with the same authentic next-token labels. The
loss is (CE(first)+CE(final))/2 at every real target. All real targets retain
unit total weight; boundary/padding exclusions and data order are unchanged.
There are no added parameters, teachers, supplied thoughts, selected-target
emphasis, stochastic routing or detached proposals. Full gradients flow
through the first softmax and probability-weighted input embedding.

Only zero and one extra pass are supported. With zero passes the objective is
exactly ordinary NTP; with one it trains both exits. A stronger same-checkpoint
deep score must not be manufactured by degrading the shallow path: retain
absolute comparisons against independently trained ordinary predictors.

CPU qualification checks old-default loss/gradient/AdamW identity, independent
SDPA/CE versus explicit attention/stable loss, causality, target masking,
full proposal gradients, exact optimizer replay and dispatched matrix products
including backward. Both objectives execute identical dense matrix work;
the extra cross-entropy reduction is not free runtime and must be measured.
The small Native stage checks all64 replicas and gradients, optimizer updates,
moments and separate-process state replay. The actual-shape stage checks local
cold gradients,100 warm updates, separate-process exact next-update replay and
a complete global256-window mature CPU gradient reference. Disposable weights
are never scientific initializations. Existing2% family and.003 loss gates,
192GiB worker allowance,512GiB host reserve and35GiB disk floors stay unchanged.

No fit is admitted by this document. Only after complete qualification and
measured cost may a separately frozen bounded screen compare the unchanged
initialization/source exposure/schedule against byte-pinned final-only and
ordinary controls. Reuse is not another seed. A positive screen would require
fresh sources/seeds and a conventional intermediate-supervision control before
attributing benefit specifically to recurrence. No automatic scaling follows.
