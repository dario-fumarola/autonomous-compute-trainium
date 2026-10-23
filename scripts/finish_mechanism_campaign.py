"""Detached finite reporting, exact-path calibration and frozen TEST confirmation."""
import argparse,json,os,subprocess,time
from pathlib import Path
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state


def main():
 p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--identity',default='post-v1');a=p.parse_args()
 a.output.mkdir(exist_ok=False);(a.output/'queue').mkdir();root=Path(os.environ['RESEARCH_ROOT']);croot=os.environ['RESEARCH_CONTAINER_ROOT'];started=time.time()
 (a.output/'plan.json').write_text(json.dumps({'identity':'mechanism-postprocessing-v1','deadline_unix':a.deadline_unix},indent=2)+'\n')
 guard_log=(a.output/'deadline-guard.log').open('w')
 subprocess.Popen([os.sys.executable,'-m','scripts.campaign_deadline','--campaign',str(a.output)],stdin=subprocess.DEVNULL,stdout=guard_log,stderr=subprocess.STDOUT,start_new_session=True)
 state={'stages':{'wait_science':{'status':'RUNNING'}},'heartbeat_unix':time.time()}
 def save():state['heartbeat_unix']=time.time();write_state(a.output/'queue/state.json',state)
 def done(status,**details):
  save();write_state(a.output/'queue/report.json',{'status':status,**details,'elapsed_seconds':time.time()-started})
 while not (a.campaign/'queue/report.json').exists():
  if time.time()>a.deadline_unix:done('POSTPROCESSING_DEADLINE');return
  save();time.sleep(10)
 science=json.loads((a.campaign/'queue/report.json').read_text());state['stages']['wait_science']['status']='COMPLETE';save()
 subprocess.run([os.sys.executable,'-m','scripts.analyze_sharing_exposure','--root',str(root),'--output','results/sharing_exposure_20261023'],check=True)
 if science['status']!='COMPLETE_QUEUE':done('TEST_BLOCKED_SCIENTIFIC_STAGE_INCOMPLETE',science_status=science['status']);return
 if time.time()+1800>a.deadline_unix:done('TEST_NOT_LAUNCHED_MEASURED_TIME_RESERVE');return
 models=[]
 for seed in [23,47,71]:
  for arm in ['ordinary','shared']:
   prefix=(f'/work/runs/natural-curve-v1-ordinary-s{seed}-u3840' if arm=='ordinary' else f'{croot}/runs/preln-recurrent-v1-s{seed}-u3840')
   models.append({'identity':f'{arm}-1b-s{seed}','arm':arm,'seed':seed,'endpoint':'1B','checkpoint':prefix+'/checkpoint.pt','reference':prefix+'-dev/report.json','passes':1 if arm=='ordinary' else 2})
 for arm,prefix,endpoint in [('untied',f'{croot}/runs/untied-depth-v1-s23-u3840','1B'),('ordinary',f'{croot}/runs/stable-cont-v2-ordinary-s23-u11520','3B'),('shared',f'{croot}/runs/stable-cont-v2-recurrent-s23-u11520','3B')]:
  models.append({'identity':f'{arm}-{endpoint.lower()}-s23','arm':arm,'seed':23,'endpoint':endpoint,'checkpoint':prefix+'/checkpoint.pt','reference':prefix+'-dev/report.json','passes':1 if arm=='ordinary' else 2})
 def execute(run,model,split,calibration=False):
  if time.time()+150>a.deadline_unix:raise TimeoutError('TEST phase budget exhausted')
  subprocess.run(['docker','update','--memory',str(224*1024**3),'autonomous-compute-neuron233'],check=True,capture_output=True)
  env=dict(os.environ,RESEARCH_RUN=run,RESEARCH_ENTRY_MODULE='scripts.evaluate_trained_path',RESEARCH_DDP_RANKS='64',
    NEURON_RT_VISIBLE_CORES=','.join(map(str,range(64))),RESEARCH_TIMEOUT_SECONDS='900')
  corpus='/work/natural-v2-1050m/dev.json' if split=='dev' else '/work/natural-v2-1050m/test-sealed.json'
  command=['python3','-m','scripts.supervise','--checkpoint',model['checkpoint'],'--corpus',corpus,'--repetitions',str(model['passes']),'--split',split]
  if calibration:command+=['--calibration-report',model['reference']]
  if split=='test':command+=['--test-release',f'{croot}/{a.output.relative_to(root)}/test-release.json']
  subprocess.run(command,env=env,check=True)
  report=json.loads((root/'runs'/run/'report.json').read_text())
  if report['status']!='PASS_SHARDED_TRAINED_PATH_READOUT':raise ValueError('Trained-path readout failed')
  return report
 try:
  state['stages']['calibration']={'status':'RUNNING'};save();calibration=[]
  for arm in ['ordinary','shared','untied']:
   model=next(m for m in models if m['arm']==arm and m['seed']==23 and m['endpoint']=='1B')
   calibration.append(execute(a.identity+'-calibration-'+arm,model,'dev',True))
  state['stages']['calibration']['status']='COMPLETE';save()
  source_files=['scripts/evaluate_trained_path.py','scripts/analyze_frozen_test.py','compute_trainium/evaluation_gate.py','compute_trainium/metrics.py','compute_trainium/training.py','compute_trainium/data.py','compute_trainium/titan.py','compute_trainium/recurrence.py','compute_trainium/preln_recurrence.py','compute_trainium/model.py']
  source={f:file_sha256(f) for f in source_files};allowed=[]
  for m in models:
   cp=(root.parent/Path(m['checkpoint']).relative_to('/work'))
   if not cp.exists():raise ValueError('Specified endpoint is missing')
   sha=file_sha256(cp);allowed.append(sha+':'+str(m['passes']));m.update(checkpoint_sha256=sha,run=a.identity+'-test-'+m['identity'])
  release={'status':'FROZEN_TEST_RELEASE','frozen_unix':time.time(),'corpus_sha256':file_sha256(root.parent/'natural-v2-1050m/test-sealed.json'),
   'allowed_checkpoint_unrolls':allowed,'models':models,'evaluator_sha256':source['scripts/evaluate_trained_path.py'],
   'analysis_sha256':source['scripts/analyze_frozen_test.py'],'source_sha256':source,'test_protocol_sha256':file_sha256('docs/FROZEN_TEST_CONFIRMATION.md')}
  write_state(a.output/'test-release.json',release);state['stages']['test']={'status':'RUNNING'};save()
  for m in models:
   for f,sha in source.items():
    if file_sha256(f)!=sha:raise ValueError('Frozen TEST evaluation/analysis source changed')
   execute(m['run'],m,'test');state['stages']['test']['last_completed']=m['identity'];save()
  state['stages']['test']['status']='COMPLETE';save()
  subprocess.run([os.sys.executable,'-m','scripts.analyze_frozen_test','--root',str(root),'--campaign',str(a.output),'--output','results/frozen_test_20261023'],check=True)
  done('COMPLETE_MECHANISM_POSTPROCESSING')
 except BaseException as e:
  done('POSTPROCESSING_FAILED',error=str(e),type=type(e).__name__);raise


if __name__=='__main__':main()
