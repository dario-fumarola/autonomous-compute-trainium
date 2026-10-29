# Sparse workspace streaming reference

This CPU-only reference retains exact per-layer K/V between calls. It supports
the same ordinary, generated-slot and trained constant-slot graphs as the active
screen. Slot placement depends on total observed source position, so arbitrary
prefill and decoding chunk sizes cannot shift a boundary. It introduces no new
thought budget, router, targets, trainable parameter or fitted model.

Tests compare every token-by-token prediction with independent full-prefix
recomputation for all three families and both attention references. Additional
checks cover chunks crossing boundaries, exact K/V inventory, isolation between
sessions, invalid inputs, changed model weights and failed-append invalidation.
All arrays are numerical software fixtures. Twelve new checks and 51 combined
checks pass in 10.21 seconds; no trained checkpoint or source document is scored.

For the current 11-layer, width-512 model at 512 source positions, the generated
and constant paths store 516 positions per layer versus ordinary's 512. At
batch one in FP32, K/V tensor payload is 23,248,896 versus 23,068,672 bytes.
That excludes model weights, allocator overhead, position arrays and temporary
concatenation. It is an inventory calculation, not measured serving memory.

No hardware speed claim follows. The reference uses Python lists and tensor
concatenation, and the Native streaming path has not been qualified. A later
hardware comparison needs exact numerical parity and actual prefill/decode
measurements on the selected trained models, including the boundary steps.
Training-update timing cannot substitute for that measurement. The active
scientific graph, budgets and quality gates remain unchanged.
