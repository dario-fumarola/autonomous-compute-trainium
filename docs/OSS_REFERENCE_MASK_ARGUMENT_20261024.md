# Optional reference mask not applied: local contribution note

Pinned MIT Parcae release69284c13746e849104f738d6d1a347b1f457df76 accepts
`attention_mask` on its model forward. Its mixer receives the argument but
calls the flash-attention interface without it. On the CPU fallback path,a
block-diagonal mask therefore has no effect and changing the supposedly hidden
prefix changes suffix outputs. This is a reproduced API limitation; whether to
implement masking or explicitly reject/document it is an upstream design decision.

Run the standalone CPU fixture with the clean pinned checkout:

```sh
USE_TORCH_XLA=0 python -m scripts.repro_reference_mask_argument \
  --upstream /path/to/parcae --output /tmp/parcae-mask-receipt.json
```

The fixture needs no private corpus,checkpoint,accelerator or generated program.
It uses48 numerical token indices in a tiny random perturbed model. It reports
output invariance only,no prediction-quality score. All installed/reference
files remain unchanged.

This does not affect the research's ordinary causal forwards,which never supply
an arbitrary attention mask. The natural-context numerical rehearsal now injects
its explicit mask at CPU SDPA in an isolated,exception-safe adapter; actual
scientific full/short scoring uses normal untouched upstream attention and
explicit original position IDs. The two initial failed rehearsals are retained.
No issue or PR has been submitted; human review is required before upstream action.
