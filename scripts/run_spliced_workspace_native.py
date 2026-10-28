"""Finite uniform-NTP independent-query Native gate,not a fit."""
import argparse,json,os,shutil,time
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from scripts.run_scale_sharding_fixture import AllChipExecution
ID='spliced-workspace-native-v1';PUBLIC=Path('results/spliced_workspace_native_20261028')


def prepare(root):
 if managed_running():raise RuntimeError('Exclusive owner required')
 ref=root/'spliced-workspace-cpu-reference-v1';receipt=json.loads((ref/'report.json').read_text());parent=json.loads((ref/'plan.json').read_text())
 if receipt['status']!='PASS_SPLICED_FULL_SIZE_CPU_REFERENCE' or file_sha256(root/receipt['gradient_relative'])!=receipt['gradient_sha256']:raise AssertionError('Full-model independent reference required')
 sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs'] for f in Path(d).rglob('*.py') if f.is_file()};sources['scripts/run_native_isolated.sh']=file_sha256('scripts/run_native_isolated.sh')
 worker,host=split_snapshot_sources(sources);projection=project_snapshot_inodes(Path('.'),1)
 if os.statvfs(root).f_favail<projection['required_free_inodes'] or shutil.disk_usage(root).free<38*1024**3:raise RuntimeError('Full qualification snapshot/artifact reserve')
 phase=ID+'-native';start=time.time();plan=dict(identity=ID,purpose='spliced_workspace_native_qualification',world=64,worker_memory_gib=192,registered_unix=start,seconds_limit=1800,phase_seconds_limit=1800,absolute_deadline_unix=start+1800,clock_admission_waiver=True,
 source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,root_comm_ports=reserve_ports(root,[phase]),phases={phase:dict(mode='native')},inode_projection=projection,
 model_config=parent['model_config'],checkpoint_relative=parent['checkpoint_relative'],checkpoint_sha256=parent['checkpoint_sha256'],gradient_relative=receipt['gradient_relative'],gradient_sha256=receipt['gradient_sha256'],inputs_relative='spliced-workspace-cpu-reference-v1/PRIVATE-inputs.pt',inputs_sha256=file_sha256(ref/'PRIVATE-inputs.pt'),cpu_reference_report_sha256=file_sha256(ref/'report.json'),planned_qualification_optimizer_updates=1,scientific_optimizer_updates=0,
 scope='New finite qualification of uniform one-target-one-CE independent-query model,32 fixed numerical queries from31..155.Independent full179-parameter gradient,actual first AdamW update/moments,and complete716-tensor/RNG perturb/restore on two shards.No large fit,allocation,teacher or quality claim;512GiB/35GiB and2% named limits unchanged.')
 out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)


def run(root):
 out=root/ID;e=AllChipExecution(root,out/'plan.json',out/'execution');p=e.plan;r=dict(status='STOPPED_SPARSE_NATIVE_QUALIFICATION',scientific_optimizer_updates=0)
 try:
  for key in ['gradient','inputs']:
   if file_sha256(root/p[key+'_relative'])!=p[key+'_sha256']:raise AssertionError('Frozen independent artifact changed')
  phase=ID+'-native';args=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+ID+'/plan.json','--plan-sha',e.sha,'--comm-port',str(p['root_comm_ports'][phase])]
  r['numerical']=e.native(phase,'scripts.qualify_sparse_workspace_native',args,'PASS_SPARSE_NATIVE_GRADIENT_UPDATE_RECOVERY');r['status']='PASS_SPARSE_NATIVE_GRADIENT_UPDATE_RECOVERY'
 except BaseException as error:r.update(error_type=type(error).__name__,error=str(error));raise
 finally:
  path=root/'runs'/(ID+'-native')/'qualification-updates.json';r['observed_qualification_updates']=json.loads(path.read_text()) if path.exists() else None
  e.close(out/'execution-costs.json');r['plan_sha256']=e.sha
  for d in [out,PUBLIC]:write_state(d/'report.json',r);write_state(d/'execution-costs.json',json.loads((out/'execution-costs.json').read_text()))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args();run(a.root) if a.run else prepare(a.root)
