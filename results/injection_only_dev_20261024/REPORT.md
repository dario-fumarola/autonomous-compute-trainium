# Same-weight boundary-injection diagnostic

On the registered12 natural DEV prefixes of seed103 (12,288 targets/model), preserving the cheap reinjection while omitting the second core remains substantially worse than the trained two-pass path.

| Model | One core | One core plus reinjection | Two cores, trained path |
| --- | ---: | ---: | ---: |
| Fixed | 3.306988 | 3.135657 | 2.613992 |
| Mixture | 2.694321 | 2.734438 | 2.658671 |

This small panel is distinct from full DEV/TEST; scores are not new endpoint results. Canonical paths agree bit-for-bit with the independent implementation. Inputs and checkpoints remain unchanged. CPU-only execution took24.36 seconds,12 configured threads; it overlaps other preparation and is not additional billed node time.

The diagnostic weakens the hypothesis that the trained model's second-pass dependence is only the boundary skip connection. Both ablated paths are post-hoc interventions. It does not establish how an independently trained cheap reinjection control would perform, or identify interpretable reasoning. No TEST, fit, source generation or model selection occurs here.

[Registration](registration.json) · [Full aggregate](report.json).
