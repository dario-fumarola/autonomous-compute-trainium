"""One paired 3B continuation, preserving the qualified full-node execution."""
import argparse,json,os,subprocess,time
from pathlib import Path
import numpy as np
from compute_trainium.training import file_sha256


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline-unix',type=float,required=True);a=p.parse_args()
 subprocess.run(['docker','start','autonomous-compute-neuron233'],check=True,capture_output=True)
 root=Path(os.environ['RESEARCH_ROOT']);old=root.parent;croot=os.environ['RESEARCH_CONTAINER_ROOT'];a.output.mkdir(exist_ok=False);started=time.time()
 data=root/'natural-append-3250m';manifest=json.loads((data/'manifest.json').read_text())
 if not (manifest['status']=='PASS_FROZEN_NATURAL_EXTENSION' and manifest['original_duplicate_index_complete'] and manifest['original_dev_test_unchanged'] and manifest['source_group_overlap']==0):
  raise ValueError('Natural extension gate incomplete')
 prefix=np.load(data/'packed/prefix_counts.npy',mmap_mode='r');targets=lambda u:int(prefix[u*256].sum())
 parents={'ordinary':'/work/runs/natural-curve-v1-ordinary-s23-u3840/stable-u3456.pt',
          'recurrent':f'{croot}/runs/preln-recurrent-v1-s23-u3456/checkpoint.pt'}
 def common(arm):
  return ['--size','small','--architecture','ordinary' if arm=='ordinary' else 'preln','--repetitions','2',
    '--context','1024','--batch','2','--accumulation','2','--precision','bf16','--optimizer','plain','--seed','23','--data-seed','23',
    '--reference-threads','8','--corpus',f'{croot}/natural-append-3250m/source.json',
    '--packed-corpus',f'{croot}/natural-append-3250m/packed','--dev-corpus','/work/natural-v2-1050m/dev.json',
    '--lr','.001' if arm=='ordinary' else '.0003','--lr-warmup-tokens',str(targets(40)),
    '--decay-start',str(targets(10368)),'--decay-end',str(targets(11520)),
    '--allow-corpus-extension','--extend-stable-schedule','--checkpoint-every','384','--warmup','4']
 # Qualify actual native recovery/data lineage on expendable copies first.
 qualification={}
 for arm in ['ordinary','recurrent']:
  name=f'continuation-v2-qual-{arm}'
  subprocess.run(['docker','update','--memory',str(288*1024**3),'autonomous-compute-neuron233'],check=True,capture_output=True)
  env=dict(os.environ,RESEARCH_RUN=name,RESEARCH_ENTRY_MODULE='scripts.train_distributed',RESEARCH_DDP_RANKS='64',
    NEURON_RT_VISIBLE_CORES=','.join(map(str,range(64))),RESEARCH_TIMEOUT_SECONDS='1200')
  subprocess.run(['python3','-m','scripts.supervise',*common(arm),'--purpose','qualification','--steps','104',
    '--pauses','0','--resume',parents[arm],'--save'],env=env,check=True)
  v=json.loads((root/'runs'/name/'report.json').read_text())
  if not v['all_replica_model_optimizer_states_equal'] or not v['recovered_model_optimizer_digest']:
   raise ValueError('Continuation full-state recovery incomplete')
  qualification[arm]=v
 (a.output/'qualification.json').write_text(json.dumps(qualification,indent=2)+'\n')
 required=sum((targets(11520)-targets(3456))/v['warm_global_real_targets_per_second'] for v in qualification.values())*1.3+6*180+600
 if time.time()+required>a.deadline_unix:raise TimeoutError('Measured complete continuation budget does not fit')
 spec={'identity':'stable-continuation-v2','seed':23,'parent_update':3456,'stable_updates':[7680,10368],
  'endpoint_update':11520,'real_targets':{str(u):targets(u) for u in [3456,7680,10368,11520]},
  'world':64,'batch':2,'accumulation':2,'optimizer':'plain','schedule':'Stable to10368, cosine1152 updates, floor0.1',
  'protocol_sha256':file_sha256('docs/MECHANISM_DIAGNOSIS_20261023.md'),
  'data_manifest_sha256':file_sha256(data/'manifest.json'),'code_sha256':file_sha256('scripts/train_distributed.py'),
  'measured_conservative_seconds':required,'deadline_unix':a.deadline_unix,'frozen_unix':time.time(),
  'scope':'Single-pair longer-exposure diagnostic; preserves original 1B negative result; no seed-variance estimate; no TEST',
  'qualification_restarted_from_original_parent_for_science':True}
 (a.output/'specification.json').write_text(json.dumps(spec,indent=2)+'\n')
 gates={k:True for k in ['immutable_sources','near_duplicate_audit','frozen_order_sha256','lr_selected_at_this_batch','measured_budget','trainer_qualified']}
 (a.output/'launch-gates.json').write_text(json.dumps(gates,indent=2)+'\n')
 records=[];previous=dict(parents);prior={'ordinary':3456,'recurrent':3456}
 for u in [7680,10368,11520]:
  for arm in ['ordinary','recurrent']:
   if file_sha256('scripts/train_distributed.py')!=spec['code_sha256']:raise ValueError('Frozen continuation source changed')
   name=f'stable-cont-v2-{arm}-s23-u{u}'
   plan={'qualification_only':False,'scientific_manifest':str(a.output/'launch-gates.json'),
    'world':64,'training_memory_gib':288,'evaluation_memory_gib':128,'timeout_seconds':7200,'evaluation_cohort':'curve',
    'resume':previous[arm],'training_arguments':[*common(arm),'--purpose','scientific'],
    'phases':[{'run':name,'steps':u-prior[arm],'pauses':0}]}
   path=a.output/(name+'-plan.json');path.write_text(json.dumps(plan,indent=2)+'\n')
   subprocess.run(['python3','-m','scripts.run_phases','--plan',str(path),'--output',str(a.output/(name+'-controller'))],check=True)
   train=json.loads((root/'runs'/name/'report.json').read_text());dev=json.loads((root/'runs'/(name+'-dev')/'report.json').read_text())
   if train['total_real_targets']!=targets(u):raise AssertionError('Matched continuation exposure differs')
   previous[arm]=f'{croot}/runs/{name}/checkpoint.pt';prior[arm]=u
   records.append({'arm':arm,'update':u,'real_tokens':train['total_real_targets'],'metrics':dev['metrics'],
    'checkpoint':previous[arm],'training_matmul_flops':train['total_training_matmul_flops'],'state_digest':train['state_digest']})
   (a.output/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
   print(json.dumps({'event':'longer_curve','arm':arm,'update':u,'nll':dev['metrics']['weighted_nll']}),flush=True)
 (a.output/'report.json').write_text(json.dumps({'status':'COMPLETE_SINGLE_PAIR_3B_CONTINUATION','specification':spec,'records':records,'elapsed_seconds':time.time()-started},indent=2)+'\n')


if __name__=='__main__':main()
