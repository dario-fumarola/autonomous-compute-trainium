"""One bounded all-chip source dependency assay;no training is admitted."""
import argparse,json,os,time,shutil
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from scripts.run_scale_sharding_fixture import AllChipExecution
ID='recoverable-dependency-inference-v1';PUBLIC=Path('results/recoverable_dependencies_20261028/inference-v1')


def prepare(root,*,recover_cache=False):
 if managed_running():raise RuntimeError('Closed prior owners required')
 panel=root/'recoverable-dependency-panel-v2';pr=json.loads((panel/'report.json').read_text());parent=json.loads((root/'spliced-workspace-discovery-v1/plan.json').read_text())
 if pr['status']!='FROZEN_RECOVERABLE_DEPENDENCY_PANEL':raise AssertionError('Adequate frozen source cohorts required')
 refs=[];reports=[]
 for identity,status in [('dependency-cpu-reference-v1','PASS_DEPENDENCY_CPU_REFERENCE'),('dependency-cpu-reference-v2','PASS_DEPENDENCY_CPU_CONTROL_REFERENCE')]:
  path=root/identity/'report.json';report=json.loads(path.read_text())
  if report['status']!=status:raise AssertionError('Independent full-size CPU forward oracle required')
  refs+=report['records'];reports.append(file_sha256(path))
 if shutil.disk_usage(root).free<37*1024**3:raise RuntimeError('2GiB measured-source/forward-artifact allowance plus35GiB reserve required')
 sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs'] for f in Path(d).rglob('*.py') if f.is_file()};sources['scripts/run_native_isolated.sh']=file_sha256('scripts/run_native_isolated.sh');worker,host=split_snapshot_sources(sources);projection=project_snapshot_inodes(Path('.'),1)
 if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('One source snapshot,compiler/artifact allowance and inode reserve required')
 for arm in ['latent','constant','ordinary']:
  audit=json.loads((root/('spliced-workspace-discovery-audit-'+arm+'-v1')/'plan.json').read_text());actual=[file_sha256(root/'runs'/parent['arm_runs'][arm]/f'rank{k}-u{parent["updates"][arm]}.zip') for k in range(64)]
  if actual!=audit['checkpoint_sha256']:raise AssertionError('Scientific endpoint changed since independent mature audit')
 start=time.time();phase=ID+'-evaluation';identity_refs=[dict(r,variant='embedding_replay') for r in refs if r.get('variant','original')=='original']
 p=dict(identity=ID,purpose='recoverable_dependency_inference',registered_unix=start,world=64,worker_memory_gib=192,seconds_limit=1800,phase_seconds_limit=1200,cpu_phase_seconds_limit=120,absolute_deadline_unix=start+1800,clock_admission_waiver=True,source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,root_comm_ports=reserve_ports(root,[phase]),phases={phase:dict(mode='frozen-source-inference')},inode_projection=projection,model_config=parent['model_config'],checkpoint_relative=parent['checkpoint_relative'],checkpoint_sha256=parent['checkpoint_sha256'],arm_runs=parent['arm_runs'],updates=parent['updates'],pause_token_id=parent['pause_token_id'],native_cache_environment=parent['native_cache_environment'],panel_relative=str((panel/'PRIVATE-cases.json').relative_to(root)),panel_sha256=pr['case_file_sha256'],source_panel_report_sha256=file_sha256(panel/'report.json'),coverage=pr['coverage'],cpu_references=refs,identity_references=identity_refs,cpu_report_sha256=reports,case_count=sum(v['cases'] for k in pr['coverage'].values() for v in k.values()),source_policy_sha256=file_sha256('docs/RECOVERABLE_DEPENDENCY_DIAGNOSTIC_20261028.md'),optimizer_updates=0,scope=Path('docs/RECOVERABLE_DEPENDENCY_DIAGNOSTIC_20261028.md').read_text())
 if recover_cache:
  failed=root/'recoverable-dependency-inference-v1';original=json.loads((failed/'plan.json').read_text());end=original['absolute_deadline_unix']
  if end-start<180:raise RuntimeError('Complete cache recovery must fit original assay deadline')
  p.update(trim_future_cache=True,cache_buckets=[32,160,1024],parent_failed_plan_sha256=file_sha256(failed/'plan.json'),diagnosis_report_sha256=file_sha256(root/'dependency-forward-diagnosis-v1/report.json'),absolute_deadline_unix=end,seconds_limit=int(end-start),phase_seconds_limit=min(1200,int(end-start)))
  p['scope']+=' Implementation-only recovery trims prefix KV strictly beyond every query position.CPU all-output/all-gradient equivalence required,all original Native tolerances and absolute deadline unchanged.No new data,mode,readout or model updates.'
 out=root/ID;out.mkdir();PUBLIC.mkdir()
 for d in [out,PUBLIC]:write_state(d/'plan.json',p)


def run(root):
 out=root/ID;e=AllChipExecution(root,out/'plan.json',out/'execution');p=e.plan;os.environ.update(p['native_cache_environment']);r=dict(status='STOPPED_RECOVERABLE_DEPENDENCY_INFERENCE',optimizer_updates=0)
 try:
  phase=ID+'-evaluation';r['evaluation']=e.native(phase,'scripts.evaluate_recoverable_dependencies',['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+ID+'/plan.json','--plan-sha',e.sha,'--comm-port',str(p['root_comm_ports'][phase])],'COMPLETE_RECOVERABLE_DEPENDENCY_INFERENCE')
  from compute_trainium.dependency_analysis import analyze
  r['analysis']=analyze(root,p);r['status']='COMPLETE_RECOVERABLE_DEPENDENCY_INFERENCE'
  for d in [out,PUBLIC]:write_state(d/'analysis.json',r['analysis'])
 except BaseException as error:r.update(error_type=type(error).__name__,error=str(error));raise
 finally:
  e.close(out/'execution-costs.json');r['plan_sha256']=e.sha
  for d in [out,PUBLIC]:write_state(d/'report.json',r);write_state(d/'execution-costs.json',json.loads((out/'execution-costs.json').read_text()))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');p.add_argument('--recover-cache',action='store_true');a=p.parse_args()
 if a.recover_cache:ID='recoverable-dependency-inference-v2';PUBLIC=Path('results/recoverable_dependencies_20261028/inference-v2')
 run(a.root) if a.run else prepare(a.root,recover_cache=a.recover_cache)
