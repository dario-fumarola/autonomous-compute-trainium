# Probability replay screen — audit recovery active

Workspace training completed200,190,896 real targets with an independent cooldown
and identical full states on64ranks. The first CPU audit stopped before gradients
or quality scoring because its digest used OrderedDict where Native's existing
cpu_tree serialization used dict. Both checkpoints reproduce the exact original
model/AdamW/RNG digest after applying that same serializer. The numeric state is
unchanged and all equality tolerances remain exact.

This recovery reuses the complete workspace and preserves every scientific
setting,remaining control endpoint and the original four-hour deadline. The
original failed CPU attempt and fit costs remain inscreen-v1. No held-out quality
result yet; no automatic promotion or larger run.
