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
python3 - <<'PY'
import hashlib,json,os,pathlib,time
root=pathlib.Path(os.environ['RESEARCH_ROOT']); run=os.environ['RESEARCH_RUN']
cores=os.environ.get('NEURON_RT_VISIBLE_CORES','0')
if '-' in cores: raise ValueError('Pass an explicit comma-separated core list')
chips=sorted({int(c)//4 for c in cores.split(',')})
p=root/'snapshots'/run
files={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*.py')}
(p/'launch.json').write_text(json.dumps({'started_unix':time.time(),'visible_cores':cores,
    'physical_chips':chips,'source_sha256':files},indent=2)+'\n')
PY
close_run() {
  RESEARCH_EXIT="$?" python3 - <<'PY'
import json,os,pathlib,time
root=pathlib.Path(os.environ['RESEARCH_ROOT']);run=os.environ['RESEARCH_RUN'];p=root/'snapshots'/run
d=json.loads((p/'launch.json').read_text());wall=time.time()-d['started_unix']
report=root/'runs'/run/'report.json'
out={'exit_code':int(os.environ['RESEARCH_EXIT']),'wall_seconds':wall,
     'assigned_chip_hours':wall*len(d['physical_chips'])/3600,
     'whole_instance_opportunity_chip_hours':wall*16/3600,
     'report_exists':report.exists()}
(p/'closure.json').write_text(json.dumps(out,indent=2)+'\n')
PY
}
trap close_run EXIT
if [[ "${RESEARCH_DDP_RANKS:-0}" -gt 0 ]]; then
  RESEARCH_ENTRY=(/work/native-venv/bin/python -m torch.distributed.run --standalone
    --nproc_per_node="$RESEARCH_DDP_RANKS" -m scripts.benchmark_distributed)
else
  RESEARCH_ENTRY=(/work/native-venv/bin/python -u -m scripts.qualify_native)
fi
docker exec \
  -e "PYTHONPATH=/work/snapshots/$RESEARCH_RUN:/work/vendor/torchtitan-neuron" \
  -e "NEURON_RT_VISIBLE_CORES=${NEURON_RT_VISIBLE_CORES:-0}" \
  -e "RESEARCH_VISIBLE_CORES=${NEURON_RT_VISIBLE_CORES:-0}" \
  -e PATH=/work/native-venv/bin:/opt/aws/neuron/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  -e LD_LIBRARY_PATH=/work/sdk/opt/aws/neuron/lib:/opt/aws/neuron/lib \
  -e TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR=/work/cache \
  -e TORCH_NEURONX_HLO_CACHE_DIR=/work/hlo-cache \
  -e TORCH_NEURONX_HLO_COMPILE_CACHE=1 \
  autonomous-compute-neuron233 \
  timeout --signal=TERM --kill-after=10s 1200 \
  "${RESEARCH_ENTRY[@]}" \
  --corpus /work/data-v1/corpus.json --output "/work/runs/$RESEARCH_RUN" "$@"
