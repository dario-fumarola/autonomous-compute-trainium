"""Bounded shared-cell screen with trained exits and earlier-state write control."""
import argparse,json,time,os,subprocess,sys,math
from pathlib import Path
from compute_trainium.training import file_sha256
from scripts.residual_memory_attribution import guard
from scripts.residual_independent_memory_training import load
ID='residual-written-memory-v1';PUBLIC=Path('results/residual_written_memory_20261028');POLICY=Path('docs/RESIDUAL_WRITTEN_MEMORY_20261028.md')


def write(out,name,value):
 for folder in [out,PUBLIC]:(folder/name).write_text(json.dumps(value,indent=2)+'\n')


def audit(root,p,path,h,y,d,head,arm,*,replay):
 import torch
 from compute_trainium.residual_local_control import make_branch
 from compute_trainium.residual_data_volume import restore_branch,source_indices
 from compute_trainium.stable_target_loss import stable_target_log_probs
 start=time.time();sha=file_sha256(path);saved=torch.load(path,weights_only=True);head64=head.double();results={};anchors=torch.arange(127,159);mode=p['branch_modes'][arm]
 for domain in range(3):
  for depth in [1,2]:
   guard(root,p['absolute_deadline_unix']);i=int((d==domain).nonzero()[0]);m=make_branch(128,mode);ref=make_branch(128,mode).double();m.load_state_dict(saved['model']);ref.load_state_dict(saved['model']);losses=[]
   for model,x,projection,reference in [(m,h[i:i+1],head,False),(ref,h[i:i+1].double(),head64,True)]:
    loss=-stable_target_log_probs(model(x,anchors,depth=depth,reference=reference)@projection.T,y[i:i+1,127:159]).mean();loss.backward();losses.append(float(loss.detach()))
   errors={name:float((v.grad.double()-dict(ref.named_parameters())[name].grad).norm()/dict(ref.named_parameters())[name].grad.norm().clamp_min(1e-30)) for name,v in m.named_parameters()}
   if any(not math.isfinite(e) or e>.02 for e in errors.values()) or abs(losses[0]-losses[1])>.0002:raise AssertionError('Supported-exit numerical gate failed')
   results[f'domain{domain}_depth{depth}']=dict(named_gradient_relative_l2=errors,loss_absolute_error=abs(losses[0]-losses[1]));del m,ref
 del head64
 if replay:
  copies=[]
  for _ in range(2):
   m=make_branch(128,mode);opt=torch.optim.AdamW(m.parameters(),lr=p['lr'],weight_decay=p['weight_decay']);rng=torch.Generator();n=restore_branch(path,m,opt,rng)
   if n!=2048:raise AssertionError('Frozen complete endpoint required')
   ix=source_indices(n,4,8192);q=torch.randperm(159,generator=rng)[:32].sort().values;loss=-stable_target_log_probs(m(h[ix],q,depth=2)@head.T,y[ix][:,q]).mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();copies.append((m.state_dict(),opt.state_dict(),rng.get_state(),float(loss.detach())))
  if copies[0][3]!=copies[1][3] or not torch.equal(copies[0][2],copies[1][2]):raise AssertionError('Restored update loss/RNG changed')
  for key,v in copies[0][0].items():
   if not torch.equal(v,copies[1][0][key]):raise AssertionError('Restored next parameter update changed')
  for index,fields in copies[0][1]['state'].items():
   for key,v in fields.items():
    if isinstance(v,torch.Tensor) and not torch.equal(v,copies[1][1]['state'][index][key]):raise AssertionError('Restored next moment update changed')
 if file_sha256(path)!=sha:raise AssertionError('Audit altered scientific checkpoint')
 return dict(status='PASS_SUPPORTED_EXITS_AUDIT',checks=results,exact_restore_and_disposable_replay=bool(replay),disposable_updates=2 if replay else 0,checkpoint_unchanged=True,wall_seconds=time.time()-start)


def prepare(root):
 import torch
 from compute_trainium.execution_isolation import managed_running
 from compute_trainium.residual_written_memory import balanced_depth_schedule
 if managed_running() or os.environ.get('USE_TORCH_XLA')!='0':raise RuntimeError('CPU-only closed Native owner required')
 start=time.time();guard(root,start+2400);torch.set_num_threads(16);out=root/ID;out.mkdir(mode=0o700);PUBLIC.mkdir();parent=root/'residual-independent-memory-v1';p=json.loads((parent/'plan.json').read_text())
 if json.loads((parent/'report.json').read_text())['status']!='COMPLETE_AUDITED_INDEPENDENT_MEMORY_TRAINING':raise AssertionError('Previous control must close first')
 p.update(identity=ID,registered_unix=start,absolute_deadline_unix=start+2400,seconds_limit=2400,cpu_phase_seconds_limit=1200,arm_modes=['static','written'],branch_modes={'static':'memory_static','written':'memory_written'},arm_unique_views={'static':8192,'written':8192},depth_schedule=balanced_depth_schedule(),source_policy_sha256=file_sha256(POLICY),scope=POLICY.read_text(),purpose='supported_iteration_with_earlier_latent_writes',retained_prefix_cost_policy='Both arms start fresh;no learned branch prefix or optimizer state reused.C0/features/old one/local fits and all failed prior histories remain shared costs.')
 for key in ['training_memory_donors','donor_mapping_receipt','source_sha256']:p.pop(key,None)
 for name,sha in p['retained_feature_sha256'].items():
  if file_sha256(root/name)!=sha:raise AssertionError('Qualified features changed')
 for ref in p['bindings'].values():
  if file_sha256(root/ref['relative'])!=ref['sha256']:raise AssertionError('Frozen comparator changed')
 if file_sha256(root/p['checkpoint_relative'])!=p['checkpoint_sha256']:raise AssertionError('C0 changed')
 reserve=Path('results/residual_memory_attribution_20261028/source-reserve.json');receipt=json.loads(reserve.read_text());reserved=root/'residual-memory-attribution-v1/PRIVATE-reserved-sources.json'
 if file_sha256(reserved)!=receipt['private_manifest_sha256']:raise AssertionError('Unscored reserve changed')
 p['reserve_receipt']=dict(private_manifest_sha256=receipt['private_manifest_sha256'],groups=320,overlap=0,scored=False,source_overlap_admission_sha256=file_sha256(parent/'admission.json'))
 h,y,d=load(root,p,'train',8192);head=torch.load(root/p['checkpoint_relative'],weights_only=True,mmap=True)['model']['lm_head.weight'].detach();cold={arm:audit(root,p,root/p['bindings']['one']['relative'],h,y,d,head,arm,replay=False) for arm in p['arm_modes']};write(out,'prefit-audit.json',cold)
 p['source_sha256']={str(f):file_sha256(f) for folder in ['compute_trainium','scripts','configs'] for f in Path(folder).rglob('*.py')};write(out,'plan.json',p);write(out,'admission.json',dict(status='PASS_WRITTEN_MEMORY_SCREEN_ADMISSION',assigned_chips=0,wall_seconds=time.time()-start,supported_depth_counts={str(i):p['depth_schedule'].count(i) for i in [1,2]},prefit_audit_passed=True));print('Prepared two supported-exit arms;no scientific updates.',flush=True)


def evaluate(root,p,out,head):
 import torch
 from compute_trainium.residual_local_control import make_branch
 from compute_trainium.residual_workspace import ResidualWorkspace
 from compute_trainium.stable_target_loss import stable_target_log_probs
 from compute_trainium.residual_screen_analysis import contrast
 from compute_trainium.dependency_analysis import interval
 test,truth,domains=load(root,p,'tune',448);models={};versions=[]
 for arm in p['arm_modes']:
  m=make_branch(128,p['branch_modes'][arm]).eval();m.load_state_dict(torch.load(out/f'PRIVATE-{arm}-state.pt',weights_only=True)['model']);m.requires_grad_(False);models[arm]=m;versions.append((m,tuple(v._version for v in m.parameters())))
 one=ResidualWorkspace(mode='one').eval();one.load_state_dict(torch.load(root/p['bindings']['one']['relative'],weights_only=True)['model']);one.requires_grad_(False);versions.append((one,tuple(v._version for v in one.parameters())));old=json.loads((root/p['bindings']['prior_one']['relative']).read_text());local=json.loads((root/p['bindings']['prior_local']['relative']).read_text());rows=[];anchors=torch.arange(127,159);max_error=0.;head_version=head._version
 with torch.no_grad():
  for i in range(448):
   guard(root,p['absolute_deadline_unix']);(out/'evaluation-progress.json').write_text(json.dumps(dict(index=i)));x=test[i:i+1];targets=truth[i:i+1,127:159];states={'raw':x[:,anchors],'independent_one':one(x,anchors)}
   for arm,m in models.items():
    for depth in [1,2]:states[f'{arm}_{depth}']=m(x,anchors,depth=depth)
   states['written_2_suppressed']=models['written'](x,anchors,depth=2,suppress_earlier_writes=True)
   row=dict(index=i,domain=int(domains[i]),group=p['evaluation_groups'][i],real_targets=32,local=local[i]['workspace_nll']);row.update({k:float(-stable_target_log_probs(v@head.T,targets).mean()) for k,v in states.items()})
   if any((v['index'],v['domain'],v['group'])!=(i,int(domains[i]),p['evaluation_groups'][i]) for v in [old[i],local[i]]):raise AssertionError('Paired source identity changed')
   err=max(abs(row['raw']-old[i]['raw_nll']),abs(row['independent_one']-old[i]['workspace_nll']),abs(row['raw']-local[i]['raw_nll']));max_error=max(max_error,err)
   if err>1e-5 or any(not math.isfinite(v) for k,v in row.items() if k!='group'):raise AssertionError('Finite score/prior parity gate failed')
   rows.append(row)
 if head._version!=head_version or any(tuple(v._version for v in m.parameters())!=version for m,version in versions):raise AssertionError('Evaluation altered model')
 (out/'PRIVATE-scores.json').write_text(json.dumps(rows));fields=['raw','local','independent_one','static_1','static_2','written_1','written_2','written_2_suppressed'];scores={k:contrast(rows,[r[k] for r in rows]) for k in fields};contrasts={}
 comparisons=[('written_two_utility','raw','written_2'),('written_two_vs_independent_one','independent_one','written_2'),('written_two_vs_own_one','written_1','written_2'),('written_two_vs_static_two','static_2','written_2'),('earlier_write_utility','written_2_suppressed','written_2'),('static_two_vs_own_one','static_1','static_2'),('written_one_vs_independent_one','independent_one','written_1')]
 for name,left,right in comparisons:contrasts[name]=contrast(rows,[r[left]-r[right] for r in rows])
 thresholds={'written_two_utility':.01,'written_two_vs_independent_one':.005,'written_two_vs_own_one':.005,'written_two_vs_static_two':.005,'earlier_write_utility':0.};gate=all(contrasts[k]['mean']>=bar and contrasts[k]['conditional_source_group_95_interval'][0]>0 for k,bar in thresholds.items());bydomain={}
 for d,name in enumerate(['web','math','code']):
  subset=[r for r in rows if r['domain']==d];bydomain[name]={k:interval(subset,[r[k] for r in subset]) for k in fields}
 return dict(scores=scores,contrasts=contrasts,domains=bydomain,useful_iteration_gate=gate,thresholds=thresholds,max_prior_score_error=max_error,reserved_groups_scored=False,scope='One development seed/reused source groups;both exits trained.No learned allocation,transfer,from-scratch,matched-total-work or confirmation claim.')


def run(root):
 import torch
 torch.set_num_threads(16);out=root/ID;path=out/'plan.json';p=json.loads(path.read_text());start=time.time();report=dict(status='STOPPED_WRITTEN_MEMORY_SCREEN',assigned_chips=0,backbone_updates=0);jobs={};logs=[]
 try:
  guard(root,p['absolute_deadline_unix'])
  for name,sha in p['source_sha256'].items():
   if file_sha256(name)!=sha:raise AssertionError('Registered source changed')
  for name,sha in p['retained_feature_sha256'].items():
   if file_sha256(root/name)!=sha:raise AssertionError('Registered features changed')
  if file_sha256(root/p['checkpoint_relative'])!=p['checkpoint_sha256']:raise AssertionError('C0 changed')
  for arm in p['arm_modes']:
   log=(out/f'{arm}-train.log').open('x');logs.append(log);jobs[arm]=subprocess.Popen([sys.executable,'-m','scripts.train_residual_screen','--root',str(root),'--plan',str(path),'--plan-sha',file_sha256(path),'--arm',arm],stdout=log,stderr=subprocess.STDOUT)
  fit_start=time.time()
  while any(job.poll() is None for job in jobs.values()):
   guard(root,min(p['absolute_deadline_unix'],fit_start+1200))
   if any(job.poll() not in [None,0] for job in jobs.values()):raise RuntimeError('A registered arm stopped;close paired controller')
   time.sleep(.5)
  if any(job.returncode for job in jobs.values()):raise RuntimeError('Scientific arm failed')
  report['arms']={arm:json.loads((out/f'{arm}-report.json').read_text()) for arm in jobs}
  if any(a['status']!='COMPLETE_RESIDUAL_SCREEN_ARM' or not a['evaluation_deferred'] for a in report['arms'].values()):raise AssertionError('Both completed unscored endpoints required')
  h,y,d=load(root,p,'train',8192);head=torch.load(root/p['checkpoint_relative'],weights_only=True,mmap=True)['model']['lm_head.weight'].detach();audits={arm:audit(root,p,out/f'PRIVATE-{arm}-state.pt',h,y,d,head,arm,replay=True) for arm in jobs};write(out,'mature-audit.json',audits);report['mature_audit']=audits;del h,y,d;report.update(evaluate(root,p,out,head))
  for ref in p['bindings'].values():
   if file_sha256(root/ref['relative'])!=ref['sha256']:raise AssertionError('Scientific comparator changed')
  report.update(status='COMPLETE_AUDITED_WRITTEN_MEMORY_SCREEN',checkpoint_sha256={arm:file_sha256(out/f'PRIVATE-{arm}-state.pt') for arm in jobs})
 except BaseException as e:report.update(error_type=type(e).__name__,error=str(e));raise
 finally:
  for job in jobs.values():
   if job.poll() is None:job.terminate()
  for job in jobs.values():
   try:job.wait(timeout=30)
   except subprocess.TimeoutExpired:job.kill();job.wait()
  for log in logs:log.close()
  report['all_arm_receipts']={arm:json.loads((out/f'{arm}-report.json').read_text()) for arm in jobs if (out/f'{arm}-report.json').exists()};report.update(wall_seconds=time.time()-start,plan_sha256=file_sha256(path));write(out,'report.json',report)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args();run(a.root) if a.run else prepare(a.root)
