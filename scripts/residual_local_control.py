"""Finite,fully accounted local-only control for the completed source-volume test."""
import argparse,json,time,os,shutil,subprocess,sys
from pathlib import Path
from compute_trainium.training import file_sha256
ID='residual-local-control-v1';PUBLIC=Path('results/residual_local_control_20261028')


def prepare(root):
 from compute_trainium.execution_isolation import managed_running
 if managed_running() or os.environ.get('USE_TORCH_XLA')!='0':raise RuntimeError('CPU-only closed Native owner required')
 available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
 if available<536*1024**3 or shutil.disk_usage(root).free<35.5*1024**3 or os.statvfs(root).f_favail<10500:raise RuntimeError('Protected reserve plus bounded CPU artifacts required')
 old=root/'residual-data-volume-v1';p=json.loads((old/'plan.json').read_text());audit=json.loads((old/'train-diagnostic.json').read_text())
 if audit['status']!='COMPLETE_RESIDUAL_TRAIN_DIAGNOSIS_AND_CPU_AUDIT' or any(not a['checkpoint_unchanged'] for a in audit['arms'].values()):raise AssertionError('Completed audited comparator required')
 start=time.time();out=root/ID;out.mkdir(mode=0o700);PUBLIC.mkdir();cache=dict(p['retained_feature_sha256'])
 for rank in range(64):
  path=root/'runs'/p['feature_phase']/f'PRIVATE-rank{rank}-train.pt';cache[str(path.relative_to(root))]=file_sha256(path)
 for name,sha in cache.items():
  if file_sha256(root/name)!=sha:raise AssertionError('Frozen feature cache changed')
 sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs'] for f in Path(d).rglob('*.py')}
 comparisons={}
 for mode in ['linear','one','two']:
  path=old/f'PRIVATE-fresh_{mode}-scores.json';comparisons[mode]=dict(relative=str(path.relative_to(root)),sha256=file_sha256(path))
 p.update(identity=ID,purpose='local_nonlinear_capacity_attribution',registered_unix=start,absolute_deadline_unix=start+1800,seconds_limit=1800,cpu_phase_seconds_limit=1200,arm_modes=['local'],branch_modes=dict(local='local'),arm_unique_views=dict(local=8192),retained_feature_sha256=cache,source_sha256=sources,comparators=comparisons,phases={},root_comm_ports={},source_policy_sha256=file_sha256('docs/RESIDUAL_LOCAL_CONTROL_20261028.md'),scope=Path('docs/RESIDUAL_LOCAL_CONTROL_20261028.md').read_text(),retained_prefix_cost_policy='Local control starts fresh;all2048 updates are new.No trained attention prefix reused.C0 and feature generation remain shared prior costs.')
 for key in ['resume_states','volume_comparison','worker_source_sha256','host_only_source_sha256','feature_extraction','inode_projection','retained_prefix_plan_sha256']:p.pop(key,None)
 for folder in [out,PUBLIC]:(folder/'plan.json').write_text(json.dumps(p,indent=2)+'\n')
 for folder in [out,PUBLIC]:(folder/'admission.json').write_text(json.dumps(dict(status='PASS_LOCAL_CONTROL_ADMISSION',assigned_chips=0,wall_seconds=time.time()-start,retained_cache_files=len(cache)),indent=2)+'\n')


def run(root):
 from compute_trainium.residual_screen_analysis import contrast
 from scripts.diagnose_residual_screen import run as audit
 out=root/ID;path=out/'plan.json';p=json.loads(path.read_text());start=time.time();report=dict(status='STOPPED_LOCAL_CONTROL',assigned_chips=0,backbone_updates=0)
 try:
  for name,sha in p['source_sha256'].items():
   if file_sha256(name)!=sha:raise AssertionError('Frozen local source changed')
  for name,sha in p['retained_feature_sha256'].items():
   if file_sha256(root/name)!=sha:raise AssertionError('Frozen features changed')
  with (out/'train.log').open('x') as log:subprocess.run([sys.executable,'-m','scripts.train_residual_screen','--root',str(root),'--plan',str(path),'--plan-sha',file_sha256(path),'--arm','local'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=min(1200,p['absolute_deadline_unix']-time.time()))
  arm=json.loads((out/'local-report.json').read_text());report.update(status='COMPLETE_FROZEN_RESIDUAL_SCREEN',arms=dict(local=arm));(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');audit(root,identity=ID,public=PUBLIC)
  rows=json.loads((out/'PRIVATE-local-scores.json').read_text());scores={key:contrast(rows,[r[key] for r in rows])['mean'] for key in ['raw_nll','workspace_nll']}
  if abs(scores['raw_nll']-2.282111260797275)>1e-5:raise AssertionError('Frozen raw score changed')
  comparisons=dict(local_utility=contrast(rows,[r['raw_nll']-r['workspace_nll'] for r in rows]))
  for mode,ref in p['comparators'].items():
   if file_sha256(root/ref['relative'])!=ref['sha256']:raise AssertionError('Frozen comparator changed')
   other=json.loads((root/ref['relative']).read_text())
   if len(rows)!=448 or any((r['index'],r['domain'],r['group'])!=(s['index'],s['domain'],s['group']) or abs(r['raw_nll']-s['raw_nll'])>1e-7 for r,s in zip(rows,other,strict=True)):raise AssertionError('Exact paired evaluation required')
   comparisons['local_minus_'+mode]=contrast(rows,[r['workspace_nll']-s['workspace_nll'] for r,s in zip(rows,other,strict=True)])
  report.update(status='COMPLETE_AUDITED_LOCAL_CONTROL',scores=scores,contrasts=comparisons,scope='Fixed after volume outcome;new control from initialization.One reused branch seed/TUNE panel.No iteration,reasoning,equivalence or efficiency claim.')
 except BaseException as e:report.update(status='STOPPED_LOCAL_CONTROL',error_type=type(e).__name__,error=str(e));raise
 finally:
  report.update(wall_seconds=time.time()-start,plan_sha256=file_sha256(path))
  for folder in [out,PUBLIC]:(folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args();run(a.root) if a.run else prepare(a.root)
