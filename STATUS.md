# Status — 2026-10-22

Fresh experiment, natural source data only. No scientific training has launched.

The predecessor pilot is closed: five qualification attempts consumed 11.8631259116 assigned chip-hours. Its hardware qualification did not pass. All predecessor jobs stopped, with all 16 chips idle after closure. The original 1.7B results are unchanged.

Implemented: natural-data preparation, domain NLL evaluation, paired source-group and three-seed statistics, small-vocabulary model definitions, TorchTitan model adapter, CPU learning diagnostic, native qualification and whole-instance throughput tools. Fourteen CPU tests pass.

Native TorchTitan ordinary pretraining passes at 150,436,608 parameters: CPU loss/gradient agreement, actual AdamW updates and exact full optimizer recovery. The 64-rank / 16-chip ordinary benchmark completes with identical model and optimizer state on every replica, measuring 157,238 real targets/sec at context 512, batch 1/core, FP32. Its cold first update takes 72.1 seconds. Tiny-model distributed gradients also match the CPU global-batch reference. DDP records an `aten::_to_copy` fallback; single-core checks record none.

The natural-text CPU diagnostic learns on all three arms: fresh DEV weighted NLL falls from approximately 11.54 to 8.09 after 64 updates. This is a 4M learning check, not a pause benefit or a converged result. TEST has not been evaluated.

Active qualification: `native-small-late-spans-001` on core 0 and `native-small-early-spans-001` on core 4, under the owned `autonomous-compute-neuron233` container. A 64-rank pause compile hit the 256 GiB container limit; it is stopped and charged. Static prediction spans now replace an expensive gather, with exact CPU output/gradient equivalence tests. Both current checks must pass before remeasuring the pause path on the whole instance and publishing the final budget.

All run sources and closures are retained under `/run/user/1000/autonomous-compute`. The original XLA environment and host driver are unchanged. See [native setup](docs/NATIVE_SETUP.md), [OSS observations](docs/OSS_NOTES.md), and [aggregate receipts](results/qualification_20261022). No large scientific run has launched.
