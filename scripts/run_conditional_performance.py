"""Bounded unscaled-fused-AdamW qualification after all science/TEST is closed."""
import argparse,json,os,subprocess,time
from pathlib import Path


def main():
 p=argparse.ArgumentParser();p.add_argument('--postprocessing',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline-unix',type=float,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
 report=a.postprocessing/'queue/report.json'
 while not report.exists():
  if time.time()>a.deadline_unix:
   (a.output/'report.json').write_text(json.dumps({'status':'PERFORMANCE_SKIPPED_DEADLINE'})+'\n');return
  time.sleep(15)
 prior=json.loads(report.read_text())
 if prior['status']!='COMPLETE_MECHANISM_POSTPROCESSING':
  (a.output/'report.json').write_text(json.dumps({'status':'PERFORMANCE_BLOCKED_PRIOR_STAGE','prior_status':prior['status']})+'\n');return
 plan=a.postprocessing/'plan.json'
 if plan.exists():a.deadline_unix=min(a.deadline_unix,json.loads(plan.read_text())['deadline_unix'])
 if time.time()+1500>a.deadline_unix:
  (a.output/'report.json').write_text(json.dumps({'status':'PERFORMANCE_SKIPPED_MEASURED_TIME_RESERVE'})+'\n');return
 started=time.time();log=(a.output/'qualification.log').open('w')
 try:
  result=subprocess.run([os.sys.executable,'-u','-m','scripts.qualify_fast_path','--output',str(a.output/'qualification'),
   '--deadline-unix',str(min(a.deadline_unix,time.time()+1200))],stdout=log,stderr=subprocess.STDOUT,timeout=1200)
  status='COMPLETE_POST_SCIENCE_PERFORMANCE_QUALIFICATION' if result.returncode==0 else 'PERFORMANCE_QUALIFICATION_FAILED'
  receipt={'status':status,'exit_code':result.returncode,'wall_seconds':time.time()-started,
   'scope':'Software/throughput qualification only, with numerical fixtures and natural TRAIN windows. No TEST selection, new research recipe or change to completed scientific models.'}
  if result.returncode==0 and time.time()+300<a.deadline_unix:
   env=dict(os.environ,RESEARCH_RUN='fused-amp-metadata-repro-001',RESEARCH_ENTRY_MODULE='scripts.repro_fused_amp_metadata',RESEARCH_DDP_RANKS='0',NEURON_RT_VISIBLE_CORES='0',RESEARCH_TIMEOUT_SECONDS='240')
   contract=subprocess.run(['python3','-m','scripts.supervise'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=270)
   receipt['optional_metadata_reproducer_exit_code']=contract.returncode
   path=Path(os.environ['RESEARCH_ROOT'])/'runs/fused-amp-metadata-repro-001/report.json'
   if path.exists():receipt['optional_metadata_reproducer_status']=json.loads(path.read_text())['status']
 except subprocess.TimeoutExpired:
  subprocess.run(['docker','stop','--time','5','autonomous-compute-neuron233'],capture_output=True,timeout=25)
  receipt={'status':'PERFORMANCE_QUALIFICATION_STOPPED_AT_BUDGET','wall_seconds':time.time()-started}
 finally:log.close()
 (a.output/'report.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
