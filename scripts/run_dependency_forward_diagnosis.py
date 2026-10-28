"""Fixed-weight diagnosis under the failed assay's original absolute cap."""
import argparse,json,os,time
from pathlib import Path
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.native_comm_ports import reserve_ports
from scripts.run_scale_sharding_fixture import AllChipExecution
ID='dependency-forward-diagnosis-v1';PUBLIC=Path('results/recoverable_dependencies_20261028/forward-diagnosis')


def prepare(root):
 if managed_running():raise RuntimeError('Failed worker must be closed')
 parent=root/'recoverable-dependency-inference-v1';p=json.loads((parent/'plan.json').read_text());r=json.loads((parent/'report.json').read_text())
 if r['status']!='STOPPED_RECOVERABLE_DEPENDENCY_INFERENCE':raise AssertionError('Preserved failed assay required')
 remaining=p['absolute_deadline_unix']-time.time()
 if remaining<180:raise RuntimeError('Diagnosis must fit original assay cap')
 sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs'] for f in Path(d).rglob('*.py') if f.is_file()};sources['scripts/run_native_isolated.sh']=file_sha256('scripts/run_native_isolated.sh');worker,host=split_snapshot_sources(sources);phase=ID+'-evaluation'
 plan=dict(p,identity=ID,registered_unix=time.time(),seconds_limit=int(remaining),phase_seconds_limit=min(600,int(remaining)),source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,root_comm_ports=reserve_ports(root,[phase]),phases={phase:dict(mode='numerical-diagnosis-only')},diagnostic_only=True,parent_failed_plan_sha256=file_sha256(parent/'plan.json'))
 out=root/ID;out.mkdir();PUBLIC.mkdir()
 for d in [out,PUBLIC]:write_state(d/'plan.json',plan)


def run(root):
 out=root/ID;e=AllChipExecution(root,out/'plan.json',out/'execution');p=e.plan;os.environ.update(p['native_cache_environment']);result=dict(status='STOPPED_DEPENDENCY_FORWARD_DIAGNOSIS',optimizer_updates=0)
 try:
  phase=ID+'-evaluation';result['native']=e.native(phase,'scripts.evaluate_recoverable_dependencies',['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+ID+'/plan.json','--plan-sha',e.sha,'--comm-port',str(p['root_comm_ports'][phase]),'--diagnose-only'],'COMPLETE_DEPENDENCY_FORWARD_DIAGNOSIS');result['status']='COMPLETE_DEPENDENCY_FORWARD_DIAGNOSIS'
 finally:
  e.close(out/'execution-costs.json')
  for d in [out,PUBLIC]:write_state(d/'report.json',result);write_state(d/'execution-costs.json',json.loads((out/'execution-costs.json').read_text()))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args();run(a.root) if a.run else prepare(a.root)
