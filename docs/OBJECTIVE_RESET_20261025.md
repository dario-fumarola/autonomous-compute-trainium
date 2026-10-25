# A bounded objective test after the small-core negative

The completed small-core pilot does not earn a larger run. Its two-pass NLL
is2.450674 at the registered training-work budget,versus2.448818 for the
parameter-exact separately trained one-pass control. Equal-data results and
the149-target natural-code readout also do not establish a useful advantage.
This is one seed on consumed development data,with all limitations retained.
The model's own useful depth response does not supersede the trained controls.

The next mechanism question is whether future-oriented supervision changes
the value of recurrent computation. Keep the already implemented fixed2×2:
one-pass versus sampled recurrence,each with ordinary next-token prediction
and with a training-only future-presence head. Future source tokens are labels;
they never enter the prefix state. No teacher,synthetic corpus,RL,new workspace,
learned halting or model scaling enters this comparison.

The precise horizon32,stride16,weight0.1,target positions t+2..t+32,source
coverage,independent rate opportunities,common full-gradient rule,cooled1B
endpoints and pilot decision rules are frozen in
[FUTURE_SPAN_OBJECTIVE_PROPOSAL_20261025.md](FUTURE_SPAN_OBJECTIVE_PROPOSAL_20261025.md).
The remaining preparatory work is numerical execution qualification,not further
architecture or objective tuning. The installed attention custom backward has
a concrete saturated-fixture discrepancy. Neither that finding nor a fix
invalidates prior fitted-model scores or proves why those comparisons lost.

A fixed finite qualification tests the explicit differentiable FP32 attention
fallback on the same numerical fixtures and mature full-batch gradients,then
actual four-cell updates,complete state/RNG recovery and100 warm varying-rate
updates. Future scientific fitting is not admitted before the complete pass
and a budget from measured throughput. Failed attempts remain in costs.

The two-seed discovery study is not terminal confirmation. Advance only if the
new objective improves recurrence's advantage by at least0.01 nats,and the
candidate also beats both same-objective one-pass and old-objective recurrence
by that amount in each pilot seed,with a useful supported depth response.
An interaction caused by damaging the one-pass auxiliary arm is insufficient.
Generic auxiliary gains can justify keeping one pass. Auxiliary loss alone
cannot justify retaining recurrence. A negative closes this particular fixed
proposal;it does not trigger a horizon/weight/depth search.

A candidate would still owe the strongest eligible ordinary/cheap-connection
baseline with the same future objective,matched training-work endpoints,
inference cost,and independent confirmation with at least five fresh paired
seeds/source groups under the existing sequential-alpha charter. Existing TEST
panels cannot become fresh by changing the objective. No confirmation is opened.

The primary papers motivate this screening decision,not a predicted victory:
[ConceptLM](https://arxiv.org/html/2602.08984v1) changes both representation and
prediction architecture;our linear presence head is not its reproduction.
[Future Summary Prediction](https://arxiv.org/html/2510.14751v2) supplies a
teacher-free multi-hot target construction,but its principal learned-summary
method uses a reverse-LM teacher and its large-scale results do not establish
this natural-only small-budget claim.
[NextLat](https://arxiv.org/html/2511.05963v3) predicts the next latent given the
next observed token;it is a different intervention,with language-model loss
and downstream metrics that need not improve together. These limitations are
reasons for controls,not reasons to assume that the objective must be the cause.
