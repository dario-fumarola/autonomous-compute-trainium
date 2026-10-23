"""Bounded untied-depth sharing control with its own equal-budget LR screen."""
import argparse,json,os,subprocess,time
from pathlib import Path
import numpy as np
from compute_trainium.training import file_sha256


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline-unix',type=float,required=True);a=p.parse_args()
 root=Path(os.environ['RESEARCH_ROOT']);croot=os.environ['RESEARCH_CONTAINER_ROOT'];a.output.mkdir(exist_ok=False);started=time.time()
 cpu=json.loads((root/'runs/untied-exact-cpu-001/report.json').read_text())
 if cpu['status']!='PASS_ACTUAL_UNTIED_FUNCTION_GRADIENT_CPU':raise ValueError('Untied CPU proof missing')
 initial_proof=json.loads((root/'runs/untied-initial-cpu-001/report.json').read_text())
 if initial_proof['status']!='PASS_FULL_SIZE_UNTIED_INITIALIZATION':raise ValueError('Full-size initial-function proof missing')
 common=['--architecture','preln','--untied-depth','--repetitions','2','--context','1024','--batch','2','--accumulation','2',
 '--precision','bf16','--optimizer','plain','--data-seed','23','--reference-threads','8',
 '--corpus','/work/natural-v2-1050m/source.json','--packed-corpus','/work/natural-v2-1050m/packed',
 '--dev-corpus','/work/natural-v2-1050m/dev.json','--pauses','0','--warmup','4','--save']
 prefix=np.load(root.parent/'natural-v2-1050m/packed/prefix_counts.npy',mmap_mode='r');targets=lambda u:int(prefix[u*256].sum())
 def run(name,arguments,cores=list(range(64)),timeout=3600):
  subprocess.run(['docker','update','--memory',str(320*1024**3),'autonomous-compute-neuron233'],check=True,capture_output=True)
  env=dict(os.environ,RESEARCH_RUN=name,RESEARCH_ENTRY_MODULE='scripts.train_distributed',RESEARCH_DDP_RANKS=str(len(cores)),
           NEURON_RT_VISIBLE_CORES=','.join(map(str,cores)),RESEARCH_TIMEOUT_SECONDS=str(timeout))
  subprocess.run(['python3','-m','scripts.supervise',*arguments],env=env,check=True)
  v=json.loads((root/'runs'/name/'report.json').read_text())
  if v['status']!='PASS_PACKED_TRAINING_PHASE':raise ValueError('Training phase failed')
  return v
 def evaluate(name):
  subprocess.run(['docker','update','--memory',str(128*1024**3),'autonomous-compute-neuron233'],check=True,capture_output=True)
  env=dict(os.environ,RESEARCH_RUN=name+'-dev',RESEARCH_EVAL='1',RESEARCH_ENTRY_MODULE='scripts.evaluate_checkpoint',RESEARCH_DDP_RANKS='1',NEURON_RT_VISIBLE_CORES='0',RESEARCH_TIMEOUT_SECONDS='900')
  subprocess.run(['python3','-m','scripts.supervise','--checkpoint',f'{croot}/runs/{name}/checkpoint.pt',
      '--corpus','/work/natural-v2-1050m/dev.json','--cohort','tune' if '-lr-' in name else 'curve'],env=env,check=True)
  return json.loads((root/'runs'/(name+'-dev')/'report.json').read_text())
 # Native numerical qualification has an explicit bounded budget.
 tiny=run('untied-tiny-ddp64-001',[*common,'--size','tiny','--seed','23','--steps','8','--check-global'])
 if not tiny['global_gradient_check']:raise ValueError('Global accumulated reference missing')
 full=run('untied-full-ddp64-001',[*common,'--size','small','--seed','23','--steps','104'])
 recovered=run('untied-restore-ddp64-001',[*common,'--size','small','--seed','23','--steps','8','--resume',f'{croot}/runs/untied-full-ddp64-001/checkpoint.pt'])
 if recovered['recovered_model_optimizer_digest']!=full['state_digest']:raise AssertionError('Untied full-state recovery differs')
 if full['initial_model_digest']!=initial_proof['full_untied_initial_digest']:raise AssertionError('Qualified untied initialization differs')
 rate=full['warm_global_real_targets_per_second']
 # Reserve the slower, fully qualified recurrent pair as well, before this fit.
 longer=(targets(3840)*2.1/834055.8+targets(3840)*2.1/604416.3)*1.3+6*180
 required=(targets(3840)+3*targets(384))/rate*1.3+6*180+longer+600
 if time.time()+required>a.deadline_unix:raise TimeoutError('Complete untied plus continuation budget does not fit')
 spec={'identity':'untied-depth-v1','protocol_sha256':file_sha256('docs/MECHANISM_DIAGNOSIS_20261023.md'),
  'code_sha256':file_sha256('scripts/train_distributed.py'),'seed':23,'tuning_seed':11,'lrs':[.0003,.001,.003],
  'parameter_count':full['parameters'],'effective_layers':36,'global_batch_windows':256,'world':64,
  'measured_rate':rate,'conservative_seconds_including_later_pair':required,'deadline_unix':a.deadline_unix,
  'parameters_matched':False,'initial_function_matched_to_shared':True,'full_initial_proof':initial_proof,'frozen_unix':time.time()}
 (a.output/'specification.json').write_text(json.dumps(spec,indent=2)+'\n')
 trials=[]
 for i,lr in enumerate(spec['lrs']):
  name=f'untied-depth-v1-lr-{i}'
  train=run(name,[*common,'--size','small','--purpose','tuning','--seed','11','--lr',str(lr),
   '--steps','384','--lr-warmup-tokens',str(targets(16)),'--decay-start',str(targets(346)),'--decay-end',str(targets(384))])
  dev=evaluate(name);trials.append({'lr':lr,'nll':dev['metrics']['weighted_nll'],'run':name})
  (a.output/'tuning.json').write_text(json.dumps(trials,indent=2)+'\n')
 selected=min(trials,key=lambda x:x['nll']);(a.output/'selected-lr.json').write_text(json.dumps({**selected,'frozen_unix':time.time()},indent=2)+'\n')
 if selected['nll']>=5:
  (a.output/'report.json').write_text(json.dumps({'status':'STOPPED_UNTIED_LEARNING_GATE','selected':selected},indent=2)+'\n');return
 previous=None;records=[];old=0
 for step in [3456,3840]:
  name=f'untied-depth-v1-s23-u{step}'
  arguments=[*common,'--size','small','--purpose','scientific','--seed','23','--lr',str(selected['lr']),
    '--lr-warmup-tokens',str(targets(40)),'--steps',str(step-old),'--checkpoint-every','384',
    '--decay-start',str(targets(3456)),'--decay-end',str(targets(3840))]
  if previous:arguments+=['--resume',previous]
  train=run(name,arguments);dev=evaluate(name)
  if train['total_real_targets']!=targets(step):raise AssertionError('Untied target exposure differs')
  if train['initial_model_digest']!=initial_proof['full_untied_initial_digest']:raise AssertionError('Scientific untied initialization differs')
  previous=f'{croot}/runs/{name}/checkpoint.pt';old=step
  records.append({'update':step,'real_tokens':train['total_real_targets'],'metrics':dev['metrics'],'train_report':train,'checkpoint':previous})
  (a.output/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
  print(json.dumps({'event':'untied_curve','update':step,'nll':dev['metrics']['weighted_nll']}),flush=True)
 (a.output/'report.json').write_text(json.dumps({'status':'COMPLETE_UNTIED_SINGLE_SEED_CONTROL','specification':spec,'records':records,'elapsed_seconds':time.time()-started},indent=2)+'\n')


if __name__=='__main__':main()
