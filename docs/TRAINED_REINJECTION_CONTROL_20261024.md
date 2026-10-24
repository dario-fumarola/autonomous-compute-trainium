# The inexpensive trained reinjection control

The fixed model differs from ordinary training in both a second core execution
and addition of its prelude state before that execution. The previous same-weight
bypass was untrained. A trained cheap control asks whether the inexpensive
connection itself can reproduce the equal-data gain.

Keep20 unique pre-norm blocks,150,436,608 parameters and the same initialization:
two prelude blocks,16 core blocks executed **once**, add the prelude state to the
result, then two coda blocks. No second core call, pass embedding, gate, new loss,
teacher or synthetic research data. Matmul estimates match ordinary; the added
elementwise work remains in wall cost. This is an architectural control, not by
itself evidence for extra computational depth.

Every checkpoint carries `reinjection_only=true` and `repetitions=1` in its
model signature. Versioned training/evaluation entries select this factory and
refuse TEST. Existing scientific forwards and frozen results remain unchanged.
The independent CPU implementation uses separate projections, attention and
RoPE. The full150M CPU numerical fixture passes at5.13e-7 relative gradient
error with zero scalar-loss difference. Its first diagnostic attempt is
preserved: a wrong flat Q/K/V gradient mapping produced an error; correcting
the checker to the actual KV-group layout changes no model function or gate.

## First execution: qualification only

Identity `reinjection-native-q-v1`, half1 after the previous fixed3B native
evaluation has closed. Other useful qualified work may occupy half0. Use the
audited6.5B natural corpus, context1024, primitive FP32 SDPA, the existing fused
AdamW recipe,32 contiguous ranks and fresh seed739. All qualification exposure
is excluded from future scientific counts.

Three bounded phases:

1. Eight actual updates at batch1/accumulation1, with the existing full global
   CPU-gradient check across all32 source windows. Keep the2% FP32 gate.
2. A fresh104-update batch1/accumulation8 execution, including100 warm timing
   updates at the intended256-window training batch. Same seed, no data wrap.
3. Eight further updates restored from the complete phase2 checkpoint, validating
   exact model/optimizer state on load and identical final replicas. No future
   trajectory bit-identity claim.

Every phase must pass its local full-gradient probe, finiteness, complete replica
state and real-token counters. The global32-window check validates the new
forward and reduction jointly; it is not represented as a full256-window trained
checkpoint qualification. A larger run would retain the existing per-phase local
probe and require its own measured budget and protocol.

Cap the sequence at25 minutes, including compilation/reference work. Record
failed attempts. No scientific sweep or pilot is launched by this qualification.
Future training should include an independently tuned FP32 ordinary comparator
with matched source order and random initialization. Reusing old BF16 ordinary
endpoints would confound the connection with precision. Keep existing consumed
TUNE/DEV for development; the fresh confirmation reserve stays unscored.
