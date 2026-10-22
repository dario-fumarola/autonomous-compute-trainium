#!/usr/bin/env bash
set -euo pipefail
RESEARCH_ROOT="${RESEARCH_ROOT:-/run/user/1000/autonomous-compute}"
RESEARCH_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESEARCH_RUN="${RESEARCH_RUN:?Set a new run identity}"
export RESEARCH_ROOT RESEARCH_RUN
python3 - <<'PY'
import os,pathlib,shutil
root=pathlib.Path(os.environ['RESEARCH_ROOT'])
for path in [root,pathlib.Path('/')]:
    if shutil.disk_usage(path).free < 35*1024**3: raise RuntimeError('Disk reserve failed')
mem=dict(line.split(':',1) for line in pathlib.Path('/proc/meminfo').read_text().splitlines())
if int(mem['MemAvailable'].split()[0])*1024 < (512+256)*1024**3: raise RuntimeError('Host memory reserve failed')
if (root/'runs'/os.environ['RESEARCH_RUN']).exists(): raise FileExistsError('Use a new run identity')
PY
mkdir -p "$RESEARCH_ROOT/snapshots/$RESEARCH_RUN"
cp -a "$RESEARCH_REPO/compute_trainium" "$RESEARCH_REPO/scripts" "$RESEARCH_REPO/configs" "$RESEARCH_ROOT/snapshots/$RESEARCH_RUN/"
docker exec \
  -e "PYTHONPATH=/work/snapshots/$RESEARCH_RUN:/work/vendor/torchtitan-neuron" \
  -e "NEURON_RT_VISIBLE_CORES=${NEURON_RT_VISIBLE_CORES:-0}" \
  -e PATH=/work/native-venv/bin:/opt/aws/neuron/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  -e LD_LIBRARY_PATH=/work/sdk/opt/aws/neuron/lib:/opt/aws/neuron/lib \
  -e TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR=/work/cache \
  autonomous-compute-neuron233 \
  timeout --signal=TERM --kill-after=10s 1200 \
  /work/native-venv/bin/python -u -m scripts.qualify_native \
  --corpus /work/data-v1/corpus.json --output "/work/runs/$RESEARCH_RUN" "$@"
