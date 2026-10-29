# Sparse persistent states match exact CPU recomputation

The CPU reference passes 35 new checks and the explicit graph factory passes
eight. The focused suite passes 76 checks; the full source suite passes 964
with two skips (159.26 seconds). Generated states, logits and all gradients match
independent full-prefix recomputation. The disabled path exactly matches the
strong tied ordinary model through an AdamW step and optimizer moments.
Prefix consistency, causal masking, cache gradients and static eager-backend
compilation pass. No research model has been trained or evaluated here.

At the prospective 51.18M shape, a slot every 128 source tokens has about 0.58%
more nominal matrix work than equally chunked ordinary execution. It makes
eight sequential body calls over a 512-token window. The arithmetic comparison
with full dense ordinary includes attention rectangles that ordinary chunking
also removes; that is not a workspace benefit. Actual hardware cost is unknown.

The active fixed-one experiment is unchanged. This reference needs separate
actual-shape numerical/recovery qualification and whole-update measurements
before any scientific training decision.

[Reference and limitations](../../../docs/CHUNKED_PERSISTENT_WORKSPACE_REFERENCE_20261029.md) ·
[Checks, hashes and all work counts](report.json).
