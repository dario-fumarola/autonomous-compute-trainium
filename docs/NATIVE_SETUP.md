# Native environment used for qualification

The host stays on Ubuntu 24.04, kernel 6.17.0-1019-aws and Neuron driver 2.30.2.0. The old XLA environment is unchanged. Native wheels need a newer glibc, so the new environment uses an isolated Ubuntu 26.04 container on the same node.

Container image: `ubuntu@sha256:c098726dd9b9733f3e885f8de0ef375206a311b0a317eadc40d347b1115fb944`. Container name: `autonomous-compute-neuron233`. Bind the new artifact directory to `/work`, expose the existing `/dev/neuron0` through `/dev/neuron15`, and mount `/opt/aws/neuron` read-only. The container has a 256 GiB memory limit and 160 CPU quota. The wrapper requires at least 768 GiB host MemAvailable before launch, so the container limit leaves the 512 GiB host reserve intact. No new instance, host driver, IAM or cloud network change is needed.

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

Keep TMPDIR and Neuron caches under the new artifact root. The wrapper snapshots the exact source for each run, rejects repeated run identities, checks disk and memory reserves, records all exit codes and assigned chip-hours, and puts an internal 20-minute timeout around the entire process tree. The runtime and compiler retain additional caches in the container; monitor the host root filesystem reserve too.

Examples from the repository root:

```sh
RESEARCH_RUN=my-new-single-core-check bash scripts/run_native.sh --size tiny --engine titan --context 128 --steps 8 --arm late --compile
RESEARCH_RUN=my-new-whole-instance-check RESEARCH_DDP_RANKS=64 NEURON_RT_VISIBLE_CORES=$(seq -s, 0 63) bash scripts/run_native.sh --size small --context 512 --steps 16 --warmup 4
```

The distributed wrapper sets one physical logical-core ID per worker before importing the backend. Inputs move explicitly to that rank's sole device. `DDP(device_ids=None)` avoids the native side-stream index inconsistency. The ordinary and pause benchmarks use the same full natural windows, initial seed, global batch, parameter count and source-token targets. Compilation, CPU reference checks, state digests, evaluation and checkpoints stay outside the steady timer and are charged separately.

The 4M correctness model is a hardware fixture. The 150M model is the smallest proposed research architecture. Neither a few updates nor a short throughput measurement establishes a scientific benefit or long-run training stability.
