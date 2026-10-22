# Reproducing the natural-text curve

Use the [frozen protocol](LEARNING_CURVE_PROTOCOL.md), [source manifest](../results/learning_curve_20261022/corpus-manifest.json), [specification](../results/learning_curve_20261022/specification.json) and [native environment](NATIVE_SETUP.md). These commands describe this instance's layout. The host controller currently names its CPU interpreter explicitly; adapt that path when reproducing elsewhere. The native model initializes inside the pinned PyTorch 2.14 container. No pretrained model weights are loaded.

## Data and qualification

The CPU preparation environment needs PyTorch, NumPy, PyArrow, tokenizers, requests and xxhash; plotting additionally needs Matplotlib. Run `python -m pytest -q` before hardware work. On the recorded instance, the CPU environment is `/home/ubuntu/ema-trainium/.venv` and pytest is supplied through `/home/ubuntu/dario/local/test-deps`.

`scripts.prepare_natural` downloads the pinned tokenizer and first raw source shards into the artifact root's `data-v1` directory. The larger preparation reuses those raw downloads, adds immutable shards, screens duplicates and freezes mapped packing:

```sh
python -m scripts.prepare_natural --output /run/user/1000/autonomous-compute/data-v1
python -m scripts.prepare_learning_corpus --output /run/user/1000/autonomous-compute/natural-v2-1050m --train-tokens 1050000000 --workers 24 --shards 6 --web-shards 2
```

Existing output identities are guarded against overwrite. Keep source content, token arrays, weights and caches outside Git. Match the published source, tokenizer, DEV, TEST and packing hashes. Train is approximately 60/20/20 by real tokens; the manifest records realized counts. DEV tuning and curve cohorts are disjoint by source group. TEST has a separate sealed file.

Complete the bounded correctness/performance gates before a scientific invocation. The controller requires the clean 100-warm-update padding qualification receipt named `learning-pad1536-qualification-001` in its artifact root. Its [exact launcher arguments](../results/learning_curve_20261022/learning-pad1536-qualification-001-launch.json), CPU comparison, throughput and replica equality are published. The [earlier trainer qualification](../results/performance_20261022/REPORT.md) supplies global-gradient, continuation and WSD replay checks. These receipts are evidence from the recorded machine; they do not replace numerical/resource checks on another environment.

The dedicated container binds the artifact root to `/work`. Plans set 256 GiB for training and 64 GiB for fresh-process DEV, in addition to the 512 GiB host reserve. Keep 35 GiB free on each output filesystem. `scripts.supervise` fails a phase on compiler-error logs, OOM events, resource exhaustion or nonzero exit; CPU agreement is independently required.

## Scientific controller

The recorded launch was:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. /home/ubuntu/ema-trainium/.venv/bin/python -u -m scripts.run_learning_experiment --data /run/user/1000/autonomous-compute/natural-v2-1050m --output /run/user/1000/autonomous-compute/natural-curve-v1 --identity natural-curve-v1
```

Use fresh output and run identities for a replication. This controller runs six independent 100M LR trials on tuning DEV, freezes the two choices, then starts fresh main models. It completes each milestone for all three paired seeds before extending any model. Training uses all 64 ranks; DEV uses a separate fresh process. Every continuation restores full model/optimizer/RNG/cursor state and checks the recovered digest before updates. The eight-hour cap and phase timeouts are encoded in the controller and plans.

Inspect `progress.json`, `tuning.json`, `selected-lrs.json`, per-phase controllers, `logs/*.supervisor.json` and immutable `snapshots/*/{launch,closure}.json` under the artifact root. Individual hardware PASS reports are not scientific benefit claims. The final primary statistic comes from the three paired endpoint NLL differences.

## Analysis and public receipts

After completion, rerun `scripts.analyze_learning_curve` to label exports complete. Run `scripts.analyze_source_groups` for separate within-model source-group uncertainty, and `scripts.verify_learning_experiment --require-complete` for all 18 curve records, hashes, initialization pairing, source-code invariance, final LR floors and clean worker receipts. Each accepts `--experiment`; the latter two also accept `--root`, and verification takes `--data`.

Create a fresh public result directory and copy only the corpus `manifest.json` there as `corpus-manifest.json`. Then run `scripts.collect_learning_experiment` with `--root`, `--experiment`, `--output`, and the inclusive `--start-utc`. Run `scripts.summarize_learning_costs` with `--root`, `--experiment`, and `--receipts` pointing to that directory; collect again to include the cost summary. Keep raw per-document scores local. The collector publishes only aggregate scores, hashes and resource receipts.

Training FLOPs are summed by phase, avoiding double counting cumulative intermediate checkpoints. Inference estimates count every actual isolated-document window, including tails, the full LM head, pauses and right padding. CPU/native correctness probes are separate model calls. These are dense matmul estimates, while worker and node times are measured. Preparation overlaps the inclusive node window and is not added twice.
