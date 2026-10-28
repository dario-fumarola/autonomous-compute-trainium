"""No-fit context-budget and content check on the original natural dependencies."""
import argparse,json,time,os
from pathlib import Path
from compute_trainium.training import file_sha256
from scripts.residual_memory_attribution import guard
ID='associative-dependency-readout-v1';PUBLIC=Path('results/associative_dependency_readout_20261028');POLICY=Path('docs/ASSOCIATIVE_DEPENDENCY_READOUT_20261028.md')


def write(out,name,value):
 for folder in [out,PUBLIC]:(folder/name).write_text(json.dumps(value,indent=2)+'\n')


def prepare(root):
 from compute_trainium.execution_isolation import managed_running
 if managed_running() or os.environ.get('USE_TORCH_XLA')!='0':raise RuntimeError('Closed Native owner;CPU only')
 start=time.time();guard(root,start+1200);out=root/ID;out.mkdir(mode=0o700);PUBLIC.mkdir();parent=json.loads((root/'causal-associative-workspace-v1/plan.json').read_text());paths=dict(c0=parent['checkpoint_relative'],cases='recoverable-dependency-panel-v2/PRIVATE-cases.json',prior='recoverable-dependency-cpu-v1/c0-metrics.json',**{arm:f'causal-associative-workspace-v1/PRIVATE-{arm}-state.pt' for arm in ['bag','cosine','learned']});bindings={k:dict(relative=v,sha256=file_sha256(root/v)) for k,v in paths.items()};cases=json.loads((root/paths['cases']).read_text());reserved=root/'residual-memory-attribution-v1/PRIVATE-reserved-sources.json'
 if file_sha256(reserved)!=parent['reserve_receipt']['private_manifest_sha256'] or {c['group'] for c in cases}&{c['group'] for c in json.loads(reserved.read_text())}:raise AssertionError('Protected reserve overlaps or changed')
 forbidden=set(parent['evaluation_groups'])
 for line in (root/'workspace-source-views-v2/train-views.jsonl').open():forbidden.add(json.loads(line)['group'])
 if any(c['group'] in forbidden for c in cases) or len(cases)!=340:raise AssertionError('Original source-isolated complete panel required')
 p=dict(identity=ID,registered_unix=start,absolute_deadline_unix=start+1200,scoring_seconds_limit=600,bindings=bindings,source_policy_sha256=file_sha256(POLICY),scope=POLICY.read_text(),source_sha256={str(f):file_sha256(f) for folder in ['compute_trainium','scripts','configs'] for f in Path(folder).rglob('*.py')},assigned_chips=0,optimizer_updates=0,base_relative='workspace-olmo2-stage1-base-v1',reserve_scored=False);write(out,'plan.json',p);write(out,'admission.json',dict(status='PASS_FROZEN_ASSOCIATIVE_READOUT_ADMISSION',cases=340,train_tune_group_overlap=0,wall_seconds=time.time()-start))


def run(root):
 import torch
 from transformers import AutoModelForCausalLM
 from compute_trainium.causal_associative_workspace import AssociativeWorkspace
 from compute_trainium.stable_target_loss import stable_target_log_probs
 from compute_trainium.dependency_analysis import interval
 out=root/ID;p=json.loads((out/'plan.json').read_text());start=time.time();deadline=min(start+600,p['absolute_deadline_unix']);torch.set_num_threads(16);report=dict(status='STOPPED_ASSOCIATIVE_DEPENDENCY_READOUT',assigned_chips=0,optimizer_updates=0)
 try:
  for name,sha in p['source_sha256'].items():
   if file_sha256(name)!=sha:raise AssertionError('Frozen source changed')
  for ref in p['bindings'].values():
   if file_sha256(root/ref['relative'])!=ref['sha256']:raise AssertionError('Frozen artifact changed')
  backbone=AutoModelForCausalLM.from_pretrained(root/p['base_relative'],local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').eval();state=torch.load(root/p['bindings']['c0']['relative'],weights_only=True,mmap=True)['model'];backbone.load_state_dict(state,assign=True);backbone.requires_grad_(False);head=backbone.lm_head.weight;models={};versions=[(backbone,tuple(v._version for v in backbone.parameters()))]
  for arm in ['bag','cosine','learned']:
   m=AssociativeWorkspace(mode=arm).eval();m.load_state_dict(torch.load(root/p['bindings'][arm]['relative'],weights_only=True)['model']);m.requires_grad_(False);models[arm]=m;versions.append((m,tuple(v._version for v in m.parameters())))
  cases=json.loads((root/p['bindings']['cases']['relative']).read_text());prior={v['index']:v for v in json.loads((root/p['bindings']['prior']['relative']).read_text()) if v['condition']=='full' and v['variant']=='original'};rows=[];max_error=0.;tick=time.time()
  with torch.no_grad():
   for i,c in enumerate(cases):
    guard(root,deadline);(out/'progress.json').write_text(json.dumps(dict(index=i,wall_seconds=time.time()-start)));tokens=torch.tensor(c['full_prefix']+c['answer'][:-1]).reshape(1,-1);labels=torch.tensor(c['full_prefix'][1:]+c['answer']).reshape(1,-1);q=torch.arange(len(c['answer']))+len(c['full_prefix'])-1;targets=torch.tensor(c['answer']).reshape(1,-1);h=backbone.model(tokens,use_cache=False).last_hidden_state;raw=stable_target_log_probs(h[:,q]@head.T,targets).double();err=max(abs(float(-raw.mean())-prior[i]['raw_nll']),abs(float(-raw[0,0])-prior[i]['raw_first_nll']));max_error=max(err,max_error)
    if err>.0002:raise AssertionError('Historical independent C0 score parity failed')
    values={'raw':raw}
    for arm,m in models.items():values[arm]=m(h,labels,q,targets,raw)
    for arm in ['cosine','learned']:values[arm+'_rotated']=models[arm](h,labels,q,targets,raw,rotate=True)
    recent=[]
    for qi,t in enumerate(q.tolist()):
     begin=max(0,t-128);recent.append(models['cosine'](h[:,begin:t+1],labels[:,begin:t+1],torch.tensor([t-begin]),targets[:,qi:qi+1],raw[:,qi:qi+1]))
    values['cosine_recent']=torch.cat(recent,1);row=dict(index=i,kind=c['kind'],cohort=c['cohort'],group=c['group'],real_targets=len(c['answer']),prefix_tokens=len(c['full_prefix']))
    for name,value in values.items():row[name]=float(-value.mean());row[name+'_first']=float(-value[0,0])
    if not all(__import__('math').isfinite(v) for k,v in row.items() if k not in ['group','kind']):raise FloatingPointError('Nonfinite readout')
    rows.append(row)
    if i==7 and time.time()+(time.time()-tick)/8*(len(cases)-8)+30>deadline:raise RuntimeError('Measured panel cannot fit frozen cap')
  if any(tuple(v._version for v in m.parameters())!=version for m,version in versions):raise AssertionError('Readout changed parameters')
  (out/'PRIVATE-scores.json').write_text(json.dumps(rows));result={};decisions={};fields=list(values)
  for kind in sorted({c['kind'] for c in rows}):
   result[kind]={};passed=[];older=[]
   for cohort in [0,1]:
    subset=[r for r in rows if r['kind']==kind and r['cohort']==cohort];scores={f:interval(subset,[r[f] for r in subset]) for f in fields};contrasts={name:interval(subset,[r[a]-r[b] for r in subset]) for name,a,b in [('cosine_utility','raw','cosine'),('learned_utility','raw','learned'),('cosine_vs_bag','bag','cosine'),('learned_vs_cosine','cosine','learned'),('cosine_association','cosine_rotated','cosine'),('learned_association','learned_rotated','learned'),('older_memory_utility','cosine_recent','cosine')]};first=[dict(r,real_targets=1) for r in subset];result[kind][str(cohort)]=dict(scores=scores,contrasts=contrasts,first_scores={f:interval(first,[r[f+'_first'] for r in subset]) for f in fields},cases=len(subset),source_groups=len({r['group'] for r in subset}));positive=lambda key:contrasts[key]['conditional_source_group_95_interval'][0]>0;passed.append(contrasts['cosine_utility']['mean']>=.03 and positive('cosine_utility') and contrasts['cosine_vs_bag']['mean']>=.01 and positive('cosine_vs_bag') and positive('cosine_association'));older.append(positive('older_memory_utility'))
   decisions[kind]=dict(exploratory_memory_lead=all(passed),primary=kind=='code_binding',older_memory_positive_in_both_cohorts=all(older))
  for ref in p['bindings'].values():
   if file_sha256(root/ref['relative'])!=ref['sha256']:raise AssertionError('Frozen artifact changed')
  report.update(status='COMPLETE_FROZEN_ASSOCIATIVE_DEPENDENCY_READOUT',families=result,decisions=decisions,max_historical_raw_error=max_error,parameters_unchanged=True,reserved_sources_scored=False,scope='Zero-update reused-panel/context-budget extrapolation;no training-seed uncertainty or reasoning/efficiency claim.')
 except BaseException as e:report.update(error_type=type(e).__name__,error=str(e));raise
 finally:report.update(wall_seconds=time.time()-start,plan_sha256=file_sha256(out/'plan.json'));write(out,'report.json',report)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args();run(a.root) if a.run else prepare(a.root)
