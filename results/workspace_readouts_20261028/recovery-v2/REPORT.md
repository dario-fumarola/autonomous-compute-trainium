# Fixed2x2 readout diagnostic

**None of the readout changes recovers a useful generated-workspace gain.** No fitting ran. Original raw and sampled/prepend scores reproduce exactly(maximum absolute difference0).

| Model | Raw next4 | Sample/prepend | Greedy/prepend | Sample/tail | Greedy/tail |
| --- | ---: | ---: | ---: | ---: | ---: |
| c0 | 2.406378 | 2.421645 | 2.430895 | 3.119288 | 3.075901 |
| workspace | 2.413154 | 2.422095 | 2.428223 | 3.139770 | 3.027175 |
| ordinary | 2.387626 | 2.395845 | 2.404517 | 3.072741 | 2.985518 |
| pause | 2.420795 | 2.437749 | 2.440646 | 3.178423 | 3.065657 |
| draft | 2.421053 | 2.429410 | 2.448026 | 3.531951 | 3.496544 |

Tail-only also hurts. At workspace weights,privileged true-future copy with tail replay reduces FIRST-target NLL to0.464140 from raw2.415373,while actual sampled/greedy notes remain harmful.That is answer-information sensitivity,not successful prefix-only writing.No generated-answer selection,reasoning,allocation or efficiency win.

All320 anchors/101 reused source groups and original checkpoints remain unchanged.All declared variants and ordinary inference controls are shown;multiple descriptive intervals are not confirmation.No new training follows from this diagnostic.

[Full result](report.json) · [Costs](execution-costs.json).
