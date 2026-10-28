#!/usr/bin/env bash
set -euo pipefail
RESEARCH_ROOT="${RESEARCH_ROOT:-/run/user/1000/autonomous-compute}"
RESEARCH_REPO="${RESEARCH_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESEARCH_RUN="${RESEARCH_RUN:?Set a new run identity}"
if [[ "${RESEARCH_LAUNCHER_FROZEN:-0}" != 1 ]]; then
  mkdir -p "$RESEARCH_ROOT/launchers"
  RESEARCH_LAUNCHER_FILE="$(mktemp "$RESEARCH_ROOT/launchers/$RESEARCH_RUN.XXXXXX.sh")"
  cp "${BASH_SOURCE[0]}" "$RESEARCH_LAUNCHER_FILE"
  export RESEARCH_REPO RESEARCH_LAUNCHER_FROZEN=1
  exec bash "$RESEARCH_LAUNCHER_FILE" "$@"
fi
RESEARCH_CONTAINER="${RESEARCH_CONTAINER:?Set an owned isolated container}"
RESEARCH_OWNER="${RESEARCH_OWNER:?Set its exact owner}"
export RESEARCH_ROOT RESEARCH_RUN RESEARCH_CONTAINER RESEARCH_OWNER
RESEARCH_CONTAINER_ROOT="${RESEARCH_CONTAINER_ROOT:-/work}"
python3 - <<'PY'
import os,pathlib,shutil,subprocess
from compute_trainium.execution_isolation import metadata,verify_owner,remaining_memory_allowance
root=pathlib.Path(os.environ['RESEARCH_ROOT'])
verify_owner(metadata(os.environ['RESEARCH_CONTAINER']),os.environ['RESEARCH_OWNER'])
for path in [root,pathlib.Path('/')]:
    if shutil.disk_usage(path).free < 35*1024**3: raise RuntimeError('Disk reserve failed')
mem=dict(line.split(':',1) for line in pathlib.Path('/proc/meminfo').read_text().splitlines())
limit=int(subprocess.check_output(['docker','inspect','--format','{{.HostConfig.Memory}}',os.environ['RESEARCH_CONTAINER']]))
if limit <= 0: raise RuntimeError('Set a positive container memory limit')
pid=subprocess.check_output(['docker','inspect','--format','{{.State.Pid}}',os.environ['RESEARCH_CONTAINER']],text=True).strip()
cg=pathlib.Path('/proc/'+pid+'/cgroup').read_text().strip().split('::',1)[1]
used=int((pathlib.Path('/sys/fs/cgroup')/cg.lstrip('/')/'memory.current').read_text())
# Existing usage is already deducted from MemAvailable. Reserve only the
# remaining shared container allowance, including when another job is active.
if int(mem['MemAvailable'].split()[0])*1024 < 512*1024**3+remaining_memory_allowance(): raise RuntimeError('Host memory reserve failed')
if (root/'runs'/os.environ['RESEARCH_RUN']).exists(): raise FileExistsError('Use a new run identity')
PY
mkdir -p "$RESEARCH_ROOT/snapshots/$RESEARCH_RUN"
cp -a "$RESEARCH_REPO/compute_trainium" "$RESEARCH_REPO/scripts" "$RESEARCH_REPO/configs" "$RESEARCH_ROOT/snapshots/$RESEARCH_RUN/"
python3 - "$@" <<'PY'
import hashlib,json,os,pathlib,time,subprocess,sys
root=pathlib.Path(os.environ['RESEARCH_ROOT']); run=os.environ['RESEARCH_RUN']
cores=os.environ.get('NEURON_RT_VISIBLE_CORES','0')
if '-' in cores: raise ValueError('Pass an explicit comma-separated core list')
chips=sorted({int(c)//4 for c in cores.split(',')})
if os.environ.get('RESEARCH_CPU_ONLY') == '1': chips=[]
p=root/'snapshots'/run
files={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*.py')}
limit=int(subprocess.check_output(['docker','inspect','--format','{{.HostConfig.Memory}}',os.environ['RESEARCH_CONTAINER']]))
(p/'launch.json').write_text(json.dumps({'started_unix':time.time(),'visible_cores':cores,'container_memory_limit_bytes':limit,'arguments':sys.argv[1:],
    'entry_module':os.environ.get('RESEARCH_ENTRY_MODULE'),
    'physical_chips':chips,'source_sha256':files,
    'native_cache_configuration':dict(hlo_eager=os.environ.get('RESEARCH_HLO_EAGER_CACHE','1'),hlo_compile=os.environ.get('RESEARCH_HLO_COMPILE_CACHE','1'),nki_compile_url=os.environ.get('RESEARCH_NKI_CACHE_URL',''))},indent=2)+'\n')
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
if [[ -n "${RESEARCH_ENTRY_MODULE:-}" ]]; then
  [[ "$RESEARCH_ENTRY_MODULE" == scripts.* ]] || exit 2
  if [[ "${RESEARCH_DDP_RANKS:-0}" -gt 1 ]]; then
    RESEARCH_ENTRY=(/work/native-venv/bin/python -m torch.distributed.run --standalone
      --nproc_per_node="$RESEARCH_DDP_RANKS" -m "$RESEARCH_ENTRY_MODULE")
  else
    RESEARCH_ENTRY=(/work/native-venv/bin/python -u -m "$RESEARCH_ENTRY_MODULE")
  fi
elif [[ "${RESEARCH_STRIDE_REPRO:-0}" -gt 0 ]]; then
  RESEARCH_ENTRY=(/work/native-venv/bin/python -u -m scripts.repro_noncontiguous_input)
elif [[ "${RESEARCH_EVAL:-0}" -gt 0 ]]; then
  RESEARCH_ENTRY=(/work/native-venv/bin/python -u -m scripts.evaluate_checkpoint)
elif [[ "${RESEARCH_TRAIN:-0}" -gt 0 ]]; then
  RESEARCH_ENTRY=(/work/native-venv/bin/python -m torch.distributed.run --standalone
    --nproc_per_node="${RESEARCH_DDP_RANKS:-1}" -m scripts.train_distributed)
elif [[ "${RESEARCH_INFERENCE_ONLY:-0}" -gt 0 ]]; then
  RESEARCH_ENTRY=(/work/native-venv/bin/python -u -m scripts.benchmark_inference)
elif [[ "${RESEARCH_DDP_RANKS:-0}" -gt 0 ]]; then
  RESEARCH_ENTRY=(/work/native-venv/bin/python -m torch.distributed.run --standalone
    --nproc_per_node="$RESEARCH_DDP_RANKS" -m scripts.benchmark_distributed)
else
  RESEARCH_ENTRY=(/work/native-venv/bin/python -u -m scripts.qualify_native)
fi
docker exec \
  -e "PYTHONPATH=$RESEARCH_CONTAINER_ROOT/snapshots/$RESEARCH_RUN:/work/vendor/torchtitan-neuron" \
  -e "NEURON_RT_VISIBLE_CORES=${NEURON_RT_VISIBLE_CORES:-0}" \
  -e "RESEARCH_VISIBLE_CORES=${NEURON_RT_VISIBLE_CORES:-0}" \
  -e PATH=/work/native-venv/bin:/opt/aws/neuron/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  -e LD_LIBRARY_PATH=/work/sdk/opt/aws/neuron/lib:/opt/aws/neuron/lib \
  -e TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR=/work/cache \
  -e TORCH_NEURONX_HLO_CACHE_DIR=/work/hlo-cache \
  -e "TORCH_NEURONX_HLO_CACHE=${RESEARCH_HLO_EAGER_CACHE:-1}" \
  -e "TORCH_NEURONX_HLO_COMPILE_CACHE=${RESEARCH_HLO_COMPILE_CACHE:-1}" \
  -e "NKI_COMPILE_CACHE_URL=${RESEARCH_NKI_CACHE_URL:-}" \
  -e TMPDIR=/work/tmp \
  "$RESEARCH_CONTAINER" \
  timeout --signal=TERM --kill-after=10s "${RESEARCH_TIMEOUT_SECONDS:-1200}" \
  "${RESEARCH_ENTRY[@]}" \
  --corpus /work/data-v1/corpus.json --output "$RESEARCH_CONTAINER_ROOT/runs/$RESEARCH_RUN" "$@"
