"""One registered training-association control;all feature/source states private."""
import argparse,json,time,os,subprocess,sys,math
from pathlib import Path
from compute_trainium.training import file_sha256
from scripts.residual_memory_attribution import guard
ID='residual-independent-memory-v1'
PUBLIC=Path('results/residual_independent_memory_20261028')
POLICY=Path('docs/RESIDUAL_INDEPENDENT_MEMORY_TRAINING_20261028.md')


def write(out,name,value):
 for folder in [out,PUBLIC]:(folder/name).write_text(json.dumps(value,indent=2)+'\n')


def load(root,p,split,count):
 import torch
 h=torch.empty(count,159,2048);y=torch.empty(count,159,dtype=torch.long);d=torch.empty(count,dtype=torch.long);seen=[]
 for phase in p['feature_segments'][split]:
  for rank in range(64):
   guard(root,p['absolute_deadline_unix']);v=torch.load(root/'runs'/phase/f'PRIVATE-rank{rank}-{split}.pt',weights_only=True,mmap=True);keep=v['indices']<count;ix=v['indices'][keep]
   h[ix]=v['features'][keep];y[ix]=v['targets'][keep];d[ix]=v['domains'][keep];seen+=ix.tolist()
 if sorted(seen)!=list(range(count)):raise AssertionError('Exact feature coverage required')
 return h,y,d


def audit(root,p,path,h,y,d,head,mapping,*,replay):
 import torch
 from compute_trainium.residual_workspace import ResidualWorkspace
 from compute_trainium.residual_independent_memory import independent_read
 from compute_trainium.residual_memory_intervention import read_with_memory
 from compute_trainium.stable_target_loss import stable_target_log_probs
 from compute_trainium.residual_data_volume import restore_branch,source_indices
 start=time.time();before=file_sha256(path);saved=torch.load(path,weights_only=True);result={};head64=head.double();anchors=torch.arange(127,159)
 for domain in range(3):
  guard(root,p['absolute_deadline_unix']);i=int((d==domain).nonzero()[0]);j=int(mapping[i]);m=ResidualWorkspace(mode='one');m.load_state_dict(saved['model']);ref=ResidualWorkspace(mode='one').double();ref.load_state_dict(saved['model']);losses=[]
  for model,features,donor,projection,fn in [(m,h[i:i+1],h[j:j+1],head,independent_read),(ref,h[i:i+1].double(),h[j:j+1].double(),head64,read_with_memory)]:
   loss=-stable_target_log_probs(fn(model,features,anchors,donor=donor)@projection.T,y[i:i+1,127:159]).mean();loss.backward();losses.append(float(loss.detach()))
  named={name:float((v.grad.double()-dict(ref.named_parameters())[name].grad).norm()/dict(ref.named_parameters())[name].grad.norm().clamp_min(1e-30)) for name,v in m.named_parameters()}
  if any(not math.isfinite(x) or x>.02 for x in named.values()) or abs(losses[0]-losses[1])>.0002:raise AssertionError('Actual donor-path FP32/FP64 numerical gate failed')
  result[str(domain)]=dict(named_gradient_relative_l2=named,loss_absolute_error=abs(losses[0]-losses[1]));del m,ref
 del head64
 if replay:
  copies=[]
  for _ in range(2):
   m=ResidualWorkspace(mode='one');opt=torch.optim.AdamW(m.parameters(),lr=p['lr'],weight_decay=p['weight_decay']);rng=torch.Generator();updates=restore_branch(path,m,opt,rng)
   if updates!=2048:raise AssertionError('Complete frozen endpoint required')
   indices=source_indices(updates,4,8192);anchors=torch.randperm(159,generator=rng)[:32].sort().values;loss=-stable_target_log_probs(independent_read(m,h[indices],anchors,h[mapping[indices]])@head.T,y[indices][:,anchors]).mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();copies.append((m.state_dict(),opt.state_dict(),rng.get_state(),float(loss.detach())))
  if copies[0][3]!=copies[1][3] or not torch.equal(copies[0][2],copies[1][2]):raise AssertionError('Exact next loss/RNG replay failed')
  for key,v in copies[0][0].items():
   if not torch.equal(v,copies[1][0][key]):raise AssertionError('Exact next model replay failed')
  for index,fields in copies[0][1]['state'].items():
   for key,v in fields.items():
    if isinstance(v,torch.Tensor) and not torch.equal(v,copies[1][1]['state'][index][key]):raise AssertionError('Exact next optimizer replay failed')
 if file_sha256(path)!=before:raise AssertionError('Audit changed scientific state')
 return dict(status='PASS_INDEPENDENT_MEMORY_NUMERICAL_AUDIT',domains=result,exact_full_state_and_next_update_replay=bool(replay),disposable_optimizer_updates=2 if replay else 0,checkpoint_unchanged=True,wall_seconds=time.time()-start)


def prepare(root):
 import torch
 from compute_trainium.execution_isolation import managed_running
 from compute_trainium.workspace_view_batches import WorkspaceViewBatches
 from compute_trainium.residual_independent_memory import source_group_derangement,fixed_evaluation_donors
 if managed_running() or os.environ.get('USE_TORCH_XLA')!='0':raise RuntimeError('Closed Native owner and CPU-only mode required')
 start=time.time();guard(root,start+1800);torch.set_num_threads(16);out=root/ID;out.mkdir(mode=0o700);PUBLIC.mkdir();p=json.loads((root/'residual-local-control-v1/plan.json').read_text())
 p.update(identity=ID,registered_unix=start,absolute_deadline_unix=start+1800,seconds_limit=1800,cpu_phase_seconds_limit=1200,arm_modes=['independent'],branch_modes={'independent':'one'},arm_unique_views={'independent':8192},evaluation_in_controller=True,source_policy_sha256=file_sha256(POLICY),scope=POLICY.read_text(),purpose='training_source_memory_association_attribution',retained_prefix_cost_policy='Fresh branch initialization;no learned branch prefix reused.C0 and all retained feature generation,original one/local fits and prior diagnostics remain separate shared research costs.')
 for key in ['resume_states','comparators','source_sha256']:p.pop(key,None)
 for name,sha in p['retained_feature_sha256'].items():
  if file_sha256(root/name)!=sha:raise AssertionError('Qualified feature cache changed')
 if file_sha256(root/p['checkpoint_relative'])!=p['checkpoint_sha256']:raise AssertionError('Frozen C0 changed')
 data=WorkspaceViewBatches(root/p['data_relative'],'train');wanted={int(row):i for i,row in enumerate(data.order[:8192])};groups=[None]*8192
 source=root/'workspace-source-views-v2/train-views.jsonl'
 for row,line in enumerate(source.open()):
  if row in wanted:groups[wanted[row]]=json.loads(line)['group']
 if any(g is None for g in groups):raise AssertionError('Training source group coverage failed')
 h,y,d=load(root,p,'train',8192);domains=d.tolist();evaldata=WorkspaceViewBatches(root/p['data_relative'],'tune');ed=[evaldata.at(i)['domain'] for i in range(448)]
 if any(a in set(p['evaluation_groups']) for a in groups):raise AssertionError('Source-separated TRAIN/TUNE required')
 reserved=root/'residual-memory-attribution-v1/PRIVATE-reserved-sources.json';receipt=json.loads(Path('results/residual_memory_attribution_20261028/source-reserve.json').read_text())
 if file_sha256(reserved)!=receipt['private_manifest_sha256']:raise AssertionError('Unscored reserve changed')
 reserve={v['group'] for v in json.loads(reserved.read_text())}
 if reserve & (set(groups)|set(p['evaluation_groups'])):raise AssertionError('Reserved group exposure forbidden')
 mapping=torch.tensor(source_group_derangement(domains,groups));evaluation=torch.tensor(fixed_evaluation_donors(domains,groups,ed,p['evaluation_groups']));map_path=out/'PRIVATE-donor-maps.pt';torch.save(dict(training=mapping,evaluation=evaluation),map_path)
 p['training_memory_donors']=dict(relative=str(map_path.relative_to(root)),sha256=file_sha256(map_path));p['donor_mapping_receipt']=dict(train_views=8192,train_groups=len(set(groups)),bijective=True,domain_counts={str(i):domains.count(i) for i in range(3)},evaluation_views=448,donors_per_evaluation_view=2,reserved_group_overlap=0,source_metadata_sha256=file_sha256(source),reserve_manifest_sha256=file_sha256(reserved))
 paths=dict(one='residual-data-volume-v1/PRIVATE-fresh_one-state.pt',prior_one='residual-data-volume-v1/PRIVATE-fresh_one-scores.json',prior_local='residual-local-control-v1/PRIVATE-local-scores.json')
 p['bindings']={name:dict(relative=rel,sha256=file_sha256(root/rel)) for name,rel in paths.items()}
 state=torch.load(root/p['checkpoint_relative'],weights_only=True,mmap=True)['model'];head=state['lm_head.weight'].detach();cold=audit(root,p,root/paths['one'],h,y,d,head,mapping,replay=False);write(out,'prefit-audit.json',cold)
 p['source_sha256']={str(f):file_sha256(f) for folder in ['compute_trainium','scripts','configs'] for f in Path(folder).rglob('*.py')};write(out,'plan.json',p);write(out,'admission.json',dict(status='PASS_INDEPENDENT_MEMORY_TRAINING_ADMISSION',assigned_chips=0,wall_seconds=time.time()-start,source_map=p['donor_mapping_receipt'],prefit_audit_passed=True));print('Prepared one independent-memory control;no scientific updates yet.',flush=True)


def evaluate(root,p,out,h,labels,domains,head,maps):
 import torch
 from compute_trainium.residual_workspace import ResidualWorkspace
 from compute_trainium.residual_independent_memory import independent_read
 from compute_trainium.stable_target_loss import stable_target_log_probs
 from compute_trainium.residual_screen_analysis import contrast
 from compute_trainium.dependency_analysis import interval
 test,truth,td=load(root,p,'tune',448);models={};versions=[]
 for name,path in [('own',root/p['bindings']['one']['relative']),('independent',out/'PRIVATE-independent-state.pt')]:
  m=ResidualWorkspace(mode='one').eval();m.load_state_dict(torch.load(path,weights_only=True)['model']);m.requires_grad_(False);models[name]=m;versions.append((m,tuple(v._version for v in m.parameters())))
 old=json.loads((root/p['bindings']['prior_one']['relative']).read_text());local=json.loads((root/p['bindings']['prior_local']['relative']).read_text());rows=[];anchors=torch.arange(127,159);max_error=0.;head_version=head._version
 with torch.no_grad():
  for i in range(448):
   guard(root,p['absolute_deadline_unix']);(out/'evaluation-progress.json').write_text(json.dumps(dict(index=i)));x=test[i:i+1];target=truth[i:i+1,127:159];row=dict(index=i,domain=int(td[i]),group=p['evaluation_groups'][i],real_targets=32)
   row['raw']=float(-stable_target_log_probs(x[:,anchors]@head.T,target).mean());row['local']=local[i]['workspace_nll']
   for name,m in models.items():
    for condition,j in [('authentic',None),('donor1',int(maps[i,0])),('donor2',int(maps[i,1]))]:
     state=m(x,anchors) if j is None else independent_read(m,x,anchors,h[j:j+1]);row[name+'_'+condition]=float(-stable_target_log_probs(state@head.T,target).mean())
   if any((v['index'],v['domain'],v['group'])!=(i,int(td[i]),p['evaluation_groups'][i]) for v in [old[i],local[i]]):raise AssertionError('Paired source identity changed')
   err=max(abs(row['own_authentic']-old[i]['workspace_nll']),abs(row['raw']-old[i]['raw_nll']),abs(row['raw']-local[i]['raw_nll']));max_error=max(err,max_error)
   if err>1e-5 or any(not math.isfinite(v) for k,v in row.items() if k not in ['group']):raise AssertionError('Frozen score parity or finite-value gate failed')
   rows.append(row)
 if head._version!=head_version or any(tuple(v._version for v in m.parameters())!=version for m,version in versions):raise AssertionError('Evaluation changed weights')
 (out/'PRIVATE-scores.json').write_text(json.dumps(rows));fields=['raw','local']+[m+'_'+c for m in models for c in ['authentic','donor1','donor2']]
 scores={k:contrast(rows,[r[k] for r in rows]) for k in fields};contrasts={'own_training_advantage':contrast(rows,[r['independent_authentic']-r['own_authentic'] for r in rows])}
 for donor in ['donor1','donor2']:
  for m in models:contrasts[m+'_'+donor+'_matching_utility']=contrast(rows,[r[m+'_'+donor]-r[m+'_authentic'] for r in rows])
  contrasts[donor+'_interaction']=contrast(rows,[(r['own_'+donor]-r['own_authentic'])-(r['independent_'+donor]-r['independent_authentic']) for r in rows])
 primary=contrasts['own_training_advantage'];positive=lambda v:v['conditional_source_group_95_interval'][0]>0
 passed=primary['mean']>=.005 and positive(primary) and all(positive(contrasts['own_'+donor+'_matching_utility']) and positive(contrasts[donor+'_interaction']) for donor in ['donor1','donor2'])
 bydomain={}
 for d,name in enumerate(['web','math','code']):
  subset=[r for r in rows if r['domain']==d];bydomain[name]={k:interval(subset,[r[k] for r in subset]) for k in fields}
 return dict(scores=scores,contrasts=contrasts,domains=bydomain,max_prior_score_error=max_error,content_learning_screen_passed=passed,reserved_groups_scored=False,scope='One reused seed/TUNE panel;conditional recipient-group intervals with fixed donor maps.No equivalence,iteration,reasoning,efficiency or confirmation claim.')


def run(root):
 import torch
 torch.set_num_threads(16);out=root/ID;path=out/'plan.json';p=json.loads(path.read_text());start=time.time();report=dict(status='STOPPED_INDEPENDENT_MEMORY_TRAINING',assigned_chips=0,backbone_updates=0)
 try:
  guard(root,p['absolute_deadline_unix'])
  for name,sha in p['source_sha256'].items():
   if file_sha256(name)!=sha:raise AssertionError('Registered source changed')
  for name,sha in p['retained_feature_sha256'].items():
   if file_sha256(root/name)!=sha:raise AssertionError('Registered feature cache changed')
  for ref in p['bindings'].values():
   if file_sha256(root/ref['relative'])!=ref['sha256']:raise AssertionError('Comparator changed')
  if file_sha256(root/p['checkpoint_relative'])!=p['checkpoint_sha256']:raise AssertionError('Frozen C0 changed')
  with (out/'train.log').open('x') as log:subprocess.run([sys.executable,'-m','scripts.train_residual_screen','--root',str(root),'--plan',str(path),'--plan-sha',file_sha256(path),'--arm','independent'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=min(1200,p['absolute_deadline_unix']-time.time()))
  arm=json.loads((out/'independent-report.json').read_text());report['arm']=arm
  if arm['status']!='COMPLETE_RESIDUAL_SCREEN_ARM' or not arm['evaluation_deferred']:raise AssertionError('Complete unaudited endpoint required before evaluation')
  h,y,d=load(root,p,'train',8192);state=torch.load(root/p['checkpoint_relative'],weights_only=True,mmap=True)['model'];head=state['lm_head.weight'].detach();ref=p['training_memory_donors']
  if file_sha256(root/ref['relative'])!=ref['sha256']:raise AssertionError('Donor maps changed')
  maps=torch.load(root/ref['relative'],weights_only=True);report['mature_audit']=audit(root,p,out/'PRIVATE-independent-state.pt',h,y,d,head,maps['training'],replay=True);write(out,'mature-audit.json',report['mature_audit']);report.update(evaluate(root,p,out,h,y,d,head,maps['evaluation']))
  for ref in p['bindings'].values():
   if file_sha256(root/ref['relative'])!=ref['sha256']:raise AssertionError('Scientific comparator changed')
  report.update(status='COMPLETE_AUDITED_INDEPENDENT_MEMORY_TRAINING',scientific_checkpoint_sha256=file_sha256(out/'PRIVATE-independent-state.pt'))
 except BaseException as e:report.update(error_type=type(e).__name__,error=str(e));raise
 finally:
  report.update(wall_seconds=time.time()-start,plan_sha256=file_sha256(path));write(out,'report.json',report)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args();run(a.root) if a.run else prepare(a.root)
