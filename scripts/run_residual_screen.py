"""Finite feature extraction and three small CPU branch fits."""
import argparse,json,os,time,subprocess,sys,signal
from pathlib import Path
from scripts.run_scale_sharding_fixture import AllChipExecution
from compute_trainium.training import file_sha256
ID='residual-workspace-screen-v1';PUBLIC=Path('results/residual_workspace_screen_20261028')


def run(root):
 out=root/ID;e=AllChipExecution(root,out/'plan.json',out/'execution');p=e.plan;os.environ.update(p['native_cache_environment']);report=dict(status='STOPPED_FROZEN_RESIDUAL_SCREEN',backbone_updates=0);jobs={};logs=[]
 try:
  for name,sha in p.get('retained_feature_sha256',{}).items():
   if file_sha256(root/name)!=sha:raise AssertionError('Retained feature cache changed')
  phase=p['feature_phase'];report['features']=e.native(phase,'scripts.extract_residual_features',['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+ID+'/plan.json','--plan-sha',e.sha,'--comm-port',str(p['root_comm_ports'][phase])],'COMPLETE_FROZEN_RESIDUAL_FEATURES')
  for arm in p['arm_modes']:
   log=open(out/f'{arm}.log','w');logs.append(log);jobs[arm]=subprocess.Popen([sys.executable,'-m','scripts.train_residual_screen','--root',str(root),'--plan',str(out/'plan.json'),'--plan-sha',e.sha,'--arm',arm],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  while any(job.poll() is None for job in jobs.values()):
   if any(job.poll() not in [None,0] for job in jobs.values()) or time.time()>p['absolute_deadline_unix']:raise RuntimeError('Frozen CPU screen worker failed/deadline')
   time.sleep(2)
  if any(job.returncode for job in jobs.values()):raise RuntimeError('CPU screen arm failed')
  if p.get('volume_comparison'):
   from compute_trainium.residual_volume_analysis import analyze
  else:
   from compute_trainium.residual_screen_analysis import analyze
  report['analysis']=analyze(out,p);report['status']='COMPLETE_FROZEN_RESIDUAL_SCREEN'
 except BaseException as error:report.update(error_type=type(error).__name__,error=str(error));raise
 finally:
  for job in jobs.values():
   if job.poll() is None:os.killpg(job.pid,signal.SIGTERM)
  for job in jobs.values():
   try:job.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);job.wait()
  for f in logs:f.close()
  e.close(out/'execution-costs.json');report['arms']={arm:json.loads((out/f'{arm}-report.json').read_text()) if (out/f'{arm}-report.json').exists() else dict(exit_code=job.returncode) for arm,job in jobs.items()}
  for folder in [out,PUBLIC]:(folder/'report.json').write_text(json.dumps(report,indent=2)+'\n');(folder/'execution-costs.json').write_text((out/'execution-costs.json').read_text())


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--volume',action='store_true');a=p.parse_args()
 if a.volume:ID='residual-data-volume-v1';PUBLIC=Path('results/residual_data_volume_20261028')
 run(a.root)
