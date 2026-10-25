# Original-work scaffold diagnostic

A CPU-only, fixed-weight readout registered after the seven foundation fits
completed and before inspecting their held-out quality. Evaluate the longer
strong-One bridge model at depth one and the R bridge model at depth two.
For every one of the same 135 DEV problems, compare the supported no-work
prompt with all its original human-written intermediate steps visible.

Keep the original premise and answer, 24-token greedy budget, EOS requirement,
whole-scalar verifier, model and tokenizer fixed. Noise depends only on the
visible prefix and is shared between models for that view. Each model's
weights are checked unchanged. Run the clean pinned upstream model entirely
in FP32 on CPU; no accelerator or optimizer updates. Bound the readout to
1,200 seconds and retain the usual host/disk reserves.

This asks whether the easiest originally scaffolded task generalizes better
than the fully hidden task. Visible human work is an **oracle intervention**;
it is not available to an autonomous prefix-only solver. A positive scaffold
effect cannot complete the research goal or reverse the running pilot's
primary comparison. The entire panel and both views are mandatory; no favorable
subset, new difficulty rule or training follow-up is selected here. Conditional
counts from one seed are descriptive, with no terminal alpha or TEST access.

## Completed result

| Fixed model | No worked steps | All original work visible |
| --- | ---: | ---: |
| Strong One, longer matched-work curriculum | 4/135 | 9/135 |
| Recurrent curriculum, declared depth two | 3/135 | 11/135 |

Showing the original steps helps somewhat, but neither model solves even 10%
of the panel with all work supplied. This weakens the premise that the easier
scaffolded task was already learned well enough to support reliable removal of
its steps. It does not uniquely diagnose curriculum failure, prove recurrence
cannot learn arithmetic or establish an autonomous-computation success.

All 135 source problems remain in both views. Final original intermediate
operation counts are {1: 109, 2: 19, 3: 7}. Numeric answers and problem text remain
private. The same complete fitted states are unchanged after generation. This
is the pinned upstream **CPU FP32** predictor; both no-work accuracy counts agree with
the completed native default-exit counts. This does not establish identical
generated tokens or argmax trajectories across precisions. No native serving-speed claim.

Wall time: **88.21 seconds**. Zero assigned chips and learning updates, no TEST,
no terminal alpha. CPU time overlaps the separate native pilot and must not be
added again to its whole-instance opportunity interval. The one-pair discovery
counts are descriptive; no independent-seed or general-text claim.
[Exact aggregates](report.json) · [Frozen specification](plan.json).
