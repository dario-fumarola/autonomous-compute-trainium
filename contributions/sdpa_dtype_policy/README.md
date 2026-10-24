# Local TorchNeuron SDPA dtype-policy proposal

This patch is **unapplied and unsubmitted**. The installed SDK and all running
experiments are unchanged. Human review and native validation are required before
any upstream submission.

The pinned backend's shared shape/dtype predicate admits BF16 and FP32, while
explicitly excluding FP16 because the NKI kernel converts its tiles to BF16.
The kernel also documents BF16 non-softmax operations for FP32 input. Our
checkpoint-free reproducer measures a full Q/K/V gradient error of0.197181 at
Q/K standard deviation8, compared with0.00001255 for the primitive FP32 route.
These are operator fixtures, not training-quality effects.

The proposed one-predicate change admits BF16 only. Both forward and backward
consult it; FP32 and FP16 use the math path. Callers seeking BF16 compute can
choose BF16 explicitly through autocast. This favors a clear dtype contract and
can slow existing FP32 callers. Upstream maintainers may prefer an explicit
accuracy/downcast policy instead; this draft does not decide that API for them.

The patch applies cleanly to a private copy of the exact installed source.
Seven CPU meta-tensor cases check both gate directions, preserve BF16 eligibility,
FP16 rejection, mixed-dtype rejection, disabled selection, explicit scale0 and
the existing length1152 behavior. The installed source hash is unchanged.
These checks cover applicability and routing predicates only. Patched native
forward/backward accuracy, compilation/cache behavior and throughput remain
unqualified. The existing primitive-route numerical measurements do not replace
that validation.

The first local preparation attempt stopped because the output parent directory
was absent, before writing a patch or touching SDK source; its elapsed CPU time
was not captured. The corrected preparation and the99-test repository suite pass.
Zero native chips were assigned to this artifact preparation.

[Unapplied patch](bf16-only-default-sdpa.patch) ·
[Local fixture report and source hashes](local-fixture-report.json) ·
[Preparation script](../../scripts/prepare_sdpa_dtype_contribution.py) ·
[Existing numerical reproducer](../../scripts/repro_sdpa_fp32_precision.py) ·
[Existing native measurements](../../results/trained_gradient_diagnosis_20261024/controls/sdpa-fp32-reproducer-report.json).
