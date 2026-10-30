# Elementary arithmetic is not yet reliable

The three completed 1B-target models do not demonstrate dependable basic
calculation on this separate source-authored diagnostic. Generated states and
ordinary each solve **13/256** original single-operation answer prefixes; trained
pauses solve **7/256**. These are the same fixed equal-data endpoints, with no
new fitting, prompting, target rewriting or model-dependent case selection.

| Model | Answer NLL | First-answer NLL | Exact answer prefixes |
|---|---:|---:|---:|
| Generated persistent state | 2.480013 | 2.288724 | 13/256 |
| Ordinary | 2.486562 | 2.327221 | 13/256 |
| Trained pauses | 2.533596 | 2.354028 | 7/256 |

Generated-state answer-NLL gain over ordinary is **0.006549**, conditional
source interval **[-0.045156, 0.056189]**. Its first-answer gain is 0.038497,
interval [-0.020955, 0.092113]. Neither establishes an advantage. Against pauses,
answer gain is 0.053583 [0.007683, 0.099428], and first-answer gain is 0.065304
[0.010618, 0.122418]. This narrower positive does not change the ordinary result.

All four operators retain equal weight and 64 original cases:

| Operator | Generated-state correct | Ordinary correct | Pause correct | Generated-state answer gain over ordinary |
|---|---:|---:|---:|---:|
| Add | 1 | 1 | 1 | +0.018965 |
| Subtract | 2 | 3 | 2 | +0.065058 |
| Multiply | 5 | 4 | 2 | -0.004549 |
| Divide | 5 | 5 | 2 | -0.053278 |

A descriptive post-readout breakdown also finds only 4/84 single-token answers
correct for both generated-state and ordinary, versus 2/84 for pauses. Thus the
low exact score cannot be attributed solely to requiring several answer tokens.
For the 172 multi-token answers, the corresponding counts are 9, 9 and 5.
This breakdown changes no threshold or selected endpoint.

The panel was frozen without model scores after a bounded scan of 47,957
original mathematical documents: 256 distinct sources and expressions, integer
operands bounded by 1,000, one original add/subtract/multiply/divide operation,
and no overlap with indexed training or protected source groups. Expression
overlap with TRAIN was deliberately **not** excluded or measured. Familiar
facts may occur; this is prerequisite competence, not novel-expression transfer.
Natural contexts and number sizes vary, so it is not a universal elementary
arithmetic benchmark. One development seed and conditional source intervals do
not measure training-seed uncertainty.

This result leaves the original broad and multi-operation outcomes unchanged.
The equal-data broad lead does not by itself establish an acquired calculation
procedure. The all-16-chip matched-work ordinary fit continues under its frozen
recipe. No architecture, extra seed, confirmation or scaling campaign follows
from this diagnostic.

Preparation takes 220.126 wall seconds; readout takes 242.123 seconds, with
three 8-thread CPU workers and peak worker RSS 1.820 GB. All nine selected/full
and batching checks pass, maximum error 4.98e-14. Independent reaggregation of
all 768 score rows agrees within 4.45e-16. Twenty-one focused tests pass. CPU
work overlaps the main Native fit and is disclosed; assigned chips and optimizer
updates are zero. Protected reserves remain unscored; terminal alpha is zero.

[Complete metrics](report.json) · [Independent reaggregation](reaggregation.json) ·
[Frozen source receipt](../source-v1/report.json) ·
[Protocol](../../../docs/PRIMITIVE_ARITHMETIC_DIAGNOSTIC_20261030.md).
