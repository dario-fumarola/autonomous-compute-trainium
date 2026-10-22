# Research instructions

Read STATUS.md and docs/PROTOCOL.md. The owner replaced the pause pilot with this fresh experiment on 2026-10-22. Use natural source text and code. No synthetic training or evaluation corpus, generated arithmetic, external teacher, or RL. Numerical arrays in correctness tests are software fixtures, never research examples. Self-generated intermediate thoughts are a possible later intervention, not a source corpus; keep them disabled in this phase.

Preserve useful-thoughts-trainium and the archived pause pilot. Do not publish private meeting transcripts, source texts/token arrays, model weights, optimizer states, caches, credentials or infrastructure identifiers. Public source code, manifests without content, and small aggregate reports are authorized. Commit and push to this public repository.

Use this existing instance. Keep 512 GiB host MemAvailable and 35 GiB free on every output filesystem. Use an isolated native TorchNeuron environment; preserve the installed XLA environment and host drivers. Include stopped/failed attempts in costs. Bound qualification attempts and record actual updates, CPU agreement, checkpoint recovery, fallbacks, compile time and steady throughput separately. Use all 16 chips when a small test is correct and resource guards pass. No large scientific training before correctness and a measured budget.

Keep real source targets, data order, initialization and parameters matched within seed. Count the vocabulary projection in FLOPs even when weights are tied. Report token-matched and FLOP-matched results separately. Track tuning and data-preparation costs. Run the tests after source changes. Update STATUS.md with what actually ran, active jobs, and next action. Do not turn a short qualification into a scientific result.

Record reproducible PyTorch/Trainium gaps in docs/OSS_NOTES.md. Prepare small local reproducers. Upstream submissions require a human review and must follow the upstream AI policy.
