# Bounded sparse persistent hardware feasibility

This is disposable numerical and execution qualification only. Finish and
preserve the active fixed-one comparison first. Neither that result nor CPU
tests automatically admits scientific training of this new mechanism.

Use the strongest existing tied ordinary shape: 11 blocks, width 512, eight
query/KV heads, FFN 1376, vocabulary 32,000, context 512 and 51,179,008 parameters.
Use one workspace slot at each full 128-source-token boundary. Freeze the
unchanged qualification TRAIN bytes, local batch two, 64 ranks/all 16 chips,
FP32 implementation and independent full-prefix CPU reference.

Qualify four explicitly named graphs:

1. Full-sequence ordinary: the existing ordinary path.
2. Chunked ordinary: exact chronological cache reuse, with no inserted slots.
3. Generated slot: the preceding normalized final state supplies each slot.
4. Constant slot: the same slot uses the mean embedding as its fixed input.

The legacy engine's `extra_steps` and `warm_by_depth` values are graph IDs here,
not latent budgets. All workspace graphs have exactly one slot per boundary.
Their model parameter inventories and initial states are identical. No teacher,
generated source corpus, held-out read, router or quality selection is used.

Before updates, compare every graph's complete actual-shape local gradients
against independent CPU execution. Preserve the existing 2% maximum family
relative-L2 and 0.003-nat scalar-loss limits. Perform 104 disposable updates,
cycling the four graphs; updates 4–103 give 25 measurements per graph, 100 in
total. Report compilation and whole-update costs separately. Require all four
mean warm times at most one second and generated/full-ordinary ratio at most
three. This is a feasibility ceiling, not a claim of efficiency. Stop early
if the registered nine-update mean exceeds two seconds.

Capture complete states at updates 103 and 104 and verify all 64 replicas.
In a fresh Native process, restore model, optimizer and RNG and reproduce the
generated-slot update 104 exactly. Compare that mature global gradient with
independent CPU full-prefix execution over all 128 windows. Constant-slot
mature gradients still require checks after any separately admitted scientific
fit; random-state qualification does not certify an entire training history.

Use a 45-minute immutable campaign deadline, 20-minute Native phase caps,
15-minute CPU audit cap, four GiB artifact allowance, projected space for two
snapshots and the unchanged 512-GiB host / 192-GiB worker / 35-GiB disk / inode
reserves. No automatic retry. Preserve initiating failures and all costs.
All qualification weights are excluded from scientific initialization.

Even a pass only supports a later frozen learning design. Such a design must
include independently tuned conventional and trained constant-slot controls,
uniform authentic-token loss, fresh scientific initialization, exact causal
readout and actual runtime accounting. It cannot claim low cost from a FLOP
ledger that gives chunking savings only to the workspace arm. Both fresh
evaluation reserves remain sealed and terminal alpha remains zero.
