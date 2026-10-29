# Bracket-type diagnostic: no clear generated-state advantage

The complete282-case,141-group diagnostic does **not** pass either registered
bracket-type gate. Once probabilities are normalized over the three closer
types and each type has equal weight, the latent model's long-target advantage
is0.040695nats over tied ordinary, conditional95% interval[-0.060128,0.139516],
and-0.002182 over trained pauses, interval[-0.102241,0.096430]. These are
inconclusive comparisons, not equivalence tests or a universal negative.

| Predictor | Long type NLL | Long type accuracy | Short type NLL | Short type accuracy |
|---|---:|---:|---:|---:|
| latent | 0.491078 | 78.84% | 0.055991 | 99.21% |
| ordinary | 0.531773 | 77.51% | 0.060755 | 98.41% |
| pause | 0.488896 | 77.12% | 0.054604 | 98.41% |

All three models benefit from full context on long cases: latent0.256618nats,
tied ordinary0.189051, pauses0.181796. Their paired additional-context advantages
remain inconclusive. Short cases are close to ceiling. Accuracy is conditional
three-class prediction, not free decoding or general code execution.

On unmodified original-target NLL, latent scores1.936364 on long cases versus
2.214491 tied ordinary and2.044123 pauses. The0.278127 gain over tied ordinary
has interval[0.081758,0.482824]; the0.107759 gain over pauses has interval
[-0.070522,0.293152]. Thus the raw-token lead does not establish improved bracket
identity tracking. The diagnostic does not prove that all gain is punctuation
calibration; raw token identity and the conditional type score are different
measurements. Neither conditional interaction clears zero.

The checkpoint was selected only by broad NLL, with exact sequential execution.
The tied challenger was independently selected on broad NLL after all three
fits/audits. Every original source target maps to its frozen vocabulary class;
no case was dropped. Two actual-weight final-only/full-readout parity checks
have zero log-probability difference. This is one seed on previously inspected
DEV, not confirmation. Fresh reserves remain closed.

Runtime1134.74s, peakRSS1.273GiB, zero chips and optimizer updates. No subsequent
fit or changed threshold is admitted by this result. The separately registered
broad conventional comparisons continue under their original criterion.
[Complete raw and typed metrics](report.json) ·
[Frozen protocol](../../../docs/INTERLEAVED_BRACKET_TYPE_20261029.md).
