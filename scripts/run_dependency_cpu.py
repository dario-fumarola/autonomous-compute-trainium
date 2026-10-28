"""Bounded CPU execution after closed Native forward failures;zero updates."""
import argparse,json,os,time,subprocess,sys,shutil
from pathlib import Path
from compute_trainium.training import file_sha256
from compute_trainium.execution_isolation import managed_running
ID='recoverable-dependency-cpu-v1';PUBLIC=Path('results/recoverable_dependencies_20261028/cpu-panel')


def run(root):
 if managed_running() or os.environ.get('USE_TORCH_XLA')!='0':raise RuntimeError('CPU only and no active Native owner required')
 start=time.time();parent=root/'recoverable-dependency-inference-v2/plan.json';p=json.loads(parent.read_text());p.update(identity=ID,registered_unix=start,absolute_deadline_unix=start+1500,seconds_limit=1500,assigned_chips=0,optimizer_updates=0,cpu_metrics_relative=ID,parent_native_plan_sha256=file_sha256(parent),cpu_execution='CPU FP32 cached graph,8 threads per model,up to4 processes;independent HF dense2D gate first;trim only future padding and independently masked unused queries;all real predictions and source criteria unchanged.',cpu_source_sha256={str(f):file_sha256(f) for d in ['compute_trainium','scripts'] for f in Path(d).rglob('*.py')})
 out=root/ID;out.mkdir(mode=0o700);PUBLIC.mkdir();jobs={};files=[];report=dict(status='STOPPED_DEPENDENCY_CPU_PANEL',optimizer_updates=0,assigned_chips=0)
 def save(name,value):
  for folder in [out,PUBLIC]:(folder/name).write_text(json.dumps(value,indent=2)+'\n')
 save('plan.json',p);sha=file_sha256(out/'plan.json')
 def launch(arm):
  log=open(out/f'{arm}.log','w');files.append(log);jobs[arm]=subprocess.Popen([sys.executable,'-m','scripts.evaluate_dependency_cpu','--root',str(root),'--plan',str(out/'plan.json'),'--plan-sha',sha,'--arm',arm],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  launch('c0')
  while not (out/'cpu-forward-gate.json').exists():
   if jobs['c0'].poll() is not None or time.time()-start>180:raise RuntimeError('C0 independent gate did not complete')
   time.sleep(1)
  for arm in ['ordinary','latent','constant']:launch(arm)
  while any(job.poll() is None for job in jobs.values()):
   failed=[arm for arm,job in jobs.items() if job.poll() not in [None,0]]
   if failed or time.time()>p['absolute_deadline_unix']:raise RuntimeError('CPU worker failure/deadline:'+str(failed))
   time.sleep(2)
  if any(job.returncode for job in jobs.values()):raise RuntimeError('CPU dependency arm failed')
  from compute_trainium.dependency_analysis import analyze
  analysis=analyze(root,p);save('analysis.json',analysis);report.update(status='COMPLETE_DEPENDENCY_CPU_PANEL',analysis_status=analysis['status'])
 finally:
  import signal
  for job in jobs.values():
   if job.poll() is None:os.killpg(job.pid,signal.SIGTERM)
  for job in jobs.values():
   try:job.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);job.wait()
  for f in files:f.close()
  report.update(wall_seconds=time.time()-start,arms={arm:json.loads((out/f'{arm}-report.json').read_text()) if (out/f'{arm}-report.json').exists() else dict(exit_code=job.returncode) for arm,job in jobs.items()},reservation_opportunity_chip_hours=(time.time()-start)*16/3600,scope='CPU-only fallback after two preserved Native numerical stops;no waived gate,no new Native attempt,no training,no confirmation.Source criteria and scientific comparisons unchanged.CPU wall and reserved-instance opportunity are distinct from zero assigned accelerator work.')
  save('report.json',report)
  if (out/'cpu-forward-gate.json').exists():save('cpu-forward-gate.json',json.loads((out/'cpu-forward-gate.json').read_text()))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
