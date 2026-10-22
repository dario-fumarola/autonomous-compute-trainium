# Native environment used for qualification

The host stays on Ubuntu 24.04, kernel 6.17.0-1019-aws and Neuron driver 2.30.2.0. The old XLA environment is unchanged. Native wheels need a newer glibc, so the new environment uses an isolated Ubuntu 26.04 container on the same node.

Container image: `ubuntu@sha256:c098726dd9b9733f3e885f8de0ef375206a311b0a317eadc40d347b1115fb944`. Container name: `autonomous-compute-neuron233`. Bind the new artifact directory to `/work`, expose the existing `/dev/neuron0` through `/dev/neuron15`, and mount `/opt/aws/neuron` read-only. The container initially had a 256 GiB memory limit and 160 CPU quota. Two pause compilation attempts hit that limit. The historical qualification temporarily used 416 GiB; the current packed trainer uses measured phase limits of 256 GiB for training and 64 GiB for DEV. The wrapper reads the actual limit and requires that amount plus 512 GiB host MemAvailable before launch. No new instance, host driver, IAM or cloud network change is needed.

Install these userspace prerequisites inside the container:

```sh
apt-get update
apt-get install -y --no-install-recommends python3.14-venv ca-certificates libnuma1 libatomic1 libgomp1 libarchive13t64 libhwloc15
python3.14 -m venv /work/native-venv
/work/native-venv/bin/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
/work/native-venv/bin/python -m pip install 'torch-neuronx==2.14.3.0.282+dc70ffa' 'neuronx-cc==2.28.7631.0+8f7b0cf5' 'nki==0.7.0+37713544100.gfede04bb' --extra-index-url https://pip.repos.neuron.amazonaws.com
```

The pinned [package inventory](../results/qualification_20261022/native-requirements.lock) records all installed dependencies, including the TorchTitan requirements and the extra Triton/torch-remat imports. It is an environment receipt, not a portable cross-platform lockfile. Clone the Neuron TorchTitan fork at `3f447e28a4cf2f8ba939024bfdd427ee8084ee8c` into `/work/vendor/torchtitan-neuron`. The wrapper adds this source tree to PYTHONPATH; it does not edit the fork.

The old host runtime lacks a symbol required by the native wheel. Download runtime `2.35.30.0-d2725ccb8` and collectives `2.35.43.0-354ad3064` from the official Neuron Ubuntu package repository. Verify their SHA256 values against the published package index and extract the packages into `/work/sdk` with `dpkg-deb -x`. Package metadata and digests are retained in the qualification reports. Put `/work/sdk/opt/aws/neuron/lib` first in LD_LIBRARY_PATH. This changes only the new container's library selection; host packages stay intact. The older host driver uses a slower supported DMA fallback, included in our timing.

Keep TMPDIR and Neuron caches under the new artifact root. The wrapper uses a shared `TORCH_NEURONX_HLO_CACHE_DIR` and enables `TORCH_NEURONX_HLO_COMPILE_CACHE=1`. It snapshots the exact source for each run, rejects repeated run identities, checks disk and memory reserves, records all exit codes and assigned chip-hours, and puts a bounded timeout around the entire process tree (20 minutes by default; the scientific plans explicitly use 30 minutes for tuning and 60 for main phases). Additional native caches were preserved and moved behind `/tmp/neuron_backend` and `/tmp/neff_cache` symlinks into `/work/compiler-debug-cache` and `/work/neff-cache`. The wrapper sets TMPDIR to `/work/tmp`. Monitor the host root filesystem reserve too.

Examples from the repository root:

```sh
docker start autonomous-compute-neuron233
RESEARCH_RUN=my-new-single-core-check bash scripts/run_native.sh --size tiny --engine titan --context 128 --steps 8 --arm late --compile
RESEARCH_RUN=my-new-whole-instance-check RESEARCH_DDP_RANKS=64 NEURON_RT_VISIBLE_CORES=$(seq -s, 0 63) bash scripts/run_native.sh --size small --context 512 --steps 16 --warmup 4
RESEARCH_RUN=my-new-inference-check RESEARCH_INFERENCE_ONLY=1 bash scripts/run_native.sh --context 512
```

Each launcher runs from its own immutable copy. Do not edit an executing shell file. Training releases unused host arenas after warmup, before the steady timer starts. Run inference in a fresh process instead of retaining training and inference graphs in all 64 workers. Stop the development container when idle; this preserves its filesystem and the bound artifact directory.

The distributed wrapper sets one physical logical-core ID per worker before importing the backend. Inputs move explicitly to that rank's sole device. `DDP(device_ids=None)` avoids the native side-stream index inconsistency. The initial benchmarks used the same full natural windows. The scientific trainer packs documents and retains every real source target, including short documents and tails. It keeps initialization, data order, global batch and parameter count matched within seed. Compilation, CPU reference checks, state digests and evaluation are charged in actual worker/node time. The original steady benchmarks exclude their final saves. Scientific phase warm timers include cadence checkpoints, including a cadence save on the final update; use complete worker time for total cost.

The 4M correctness model is a hardware fixture. The 150M model is the smallest proposed research architecture. Neither a few updates nor a short throughput measurement establishes a scientific benefit or long-run training stability.

The active natural-text campaign and its exact invocation are described in [Reproducing the curve](REPRODUCE_CURVE.md).
