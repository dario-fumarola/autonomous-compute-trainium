"""Frozen-source learned associative-memory screen with strong cache controls."""
import argparse,json,time,os,sys,subprocess,math
from pathlib import Path
from compute_trainium.training import file_sha256
from scripts.residual_memory_attribution import guard
from scripts.residual_independent_memory_training import load
ID='causal-associative-workspace-v1';PUBLIC=Path('results/causal_associative_workspace_20261028');POLICY=Path('docs/CAUSAL_ASSOCIATIVE_WORKSPACE_20261028.md')


def write(out,name,value):
 for folder in [out,PUBLIC]:(folder/name).write_text(json.dumps(value,indent=2)+'\n')


def qualify(root,p,h,y,d,head,*,saved=None,mode='learned'):
 import torch
 from compute_trainium.causal_associative_workspace import AssociativeWorkspace,sequential_reference
 from compute_trainium.stable_target_loss import stable_target_log_probs
 start=time.time();results={};q=torch.arange(127,159)
 for domain in range(3):
  guard(root,p['absolute_deadline_unix']);i=int((d==domain).nonzero()[0]);m=AssociativeWorkspace(mode=mode);ref=AssociativeWorkspace(mode=mode).double()
  if saved:m.load_state_dict(saved['model'])
  else:m.strength.data.fill_(.1)
  ref.load_state_dict(m.state_dict());raw=stable_target_log_probs(h[i:i+1,q]@head.T,y[i:i+1,q]).detach();loss=-m(h[i:i+1],y[i:i+1],q,y[i:i+1,q],raw).mean();other=-sequential_reference(ref,h[i:i+1].double(),y[i:i+1],q,y[i:i+1,q],raw.double()).mean();loss.backward();other.backward();errors={name:float((v.grad.double()-dict(ref.named_parameters())[name].grad).norm()/dict(ref.named_parameters())[name].grad.norm().clamp_min(1e-30)) for name,v in m.named_parameters()}
  if any(not math.isfinite(e) or e>.02 for e in errors.values()) or abs(float(loss.detach()-other.detach()))>.0002:raise AssertionError('Independent sequential-reference numerical gate failed')
  results[str(domain)]=dict(named_gradient_relative_l2=errors,loss_error=abs(float(loss.detach()-other.detach())))
 return dict(status='PASS_ASSOCIATIVE_NUMERICAL_AUDIT',domains=results,wall_seconds=time.time()-start)


def prepare(root):
 import torch
 from compute_trainium.execution_isolation import managed_running
 from compute_trainium.workspace_view_batches import WorkspaceViewBatches
 if managed_running() or os.environ.get('USE_TORCH_XLA')!='0':raise RuntimeError('Closed owner and CPU mode required')
 start=time.time();torch.set_num_threads(16);guard(root,start+2700);out=root/ID;out.mkdir(mode=0o700);PUBLIC.mkdir();parent=root/'residual-written-memory-v1';p=json.loads((parent/'plan.json').read_text());data=WorkspaceViewBatches(root/p['data_relative'],'train');wanted={int(row):i for i,row in enumerate(data.order[:8192])};groups=[None]*8192
 for row,line in enumerate((root/'workspace-source-views-v2/train-views.jsonl').open()):
  if row in wanted:groups[wanted[row]]=json.loads(line)['group']
 if any(g is None for g in groups):raise AssertionError('Source group coverage required')
 cal=[i for i,g in enumerate(groups) if int(g[:8],16)%8==0];fit=[i for i,g in enumerate(groups) if int(g[:8],16)%8!=0][:6144]
 if len(fit)!=6144 or len(cal)!=797 or set(groups[i] for i in fit)&set(groups[i] for i in cal):raise AssertionError('Frozen source split failed')
 reserved=root/'residual-memory-attribution-v1/PRIVATE-reserved-sources.json'
 if file_sha256(reserved)!=p['reserve_receipt']['private_manifest_sha256'] or {v['group'] for v in json.loads(reserved.read_text())}&(set(groups)|set(p['evaluation_groups'])):raise AssertionError('Protected source reserve changed or overlaps')
 p.update(identity=ID,registered_unix=start,absolute_deadline_unix=start+2700,seconds_limit=2700,seed=8317,query_seed=8318,updates=1536,fit_views=6144,calibration_views=797,arm_modes=['bag','cosine','learned'],source_policy_sha256=file_sha256(POLICY),scope=POLICY.read_text(),temperatures=[0,4,16,64,256],mixture_weights=[0,.005,.01,.03,.1,.3,.5],purpose='causal_observed_outcome_memory',retained_prefix_cost_policy='New branch fits only;C0,feature generation and all previous comparator fits remain shared research costs.')
 for key in ['depth_schedule','branch_modes','arm_unique_views','source_sha256']:p.pop(key,None)
 for name,sha in p['retained_feature_sha256'].items():
  if file_sha256(root/name)!=sha:raise AssertionError('Frozen feature cache changed')
 if file_sha256(root/p['checkpoint_relative'])!=p['checkpoint_sha256']:raise AssertionError('C0 changed')
 maps=dict(fit=torch.tensor(fit),calibration=torch.tensor(cal));path=out/'PRIVATE-source-map.pt';torch.save(maps,path);p['source_map']=dict(relative=str(path.relative_to(root)),sha256=file_sha256(path),fit_groups=len({groups[i] for i in fit}),calibration_groups=len({groups[i] for i in cal}),group_overlap=0,reserve_overlap=0)
 h,y,d=load(root,p,'train',8192);head=torch.load(root/p['checkpoint_relative'],weights_only=True,mmap=True)['model']['lm_head.weight'].detach();torch.manual_seed(8317);audits={arm:qualify(root,p,h,y,d,head,mode=arm) for arm in p['arm_modes']};write(out,'prefit-audit.json',audits);p['source_sha256']={str(f):file_sha256(f) for folder in ['compute_trainium','scripts','configs'] for f in Path(folder).rglob('*.py')};write(out,'plan.json',p);write(out,'admission.json',dict(status='PASS_CAUSAL_ASSOCIATIVE_ADMISSION',wall_seconds=time.time()-start,assigned_chips=0,source_split=p['source_map']));print('Prepared causal-memory screen;no optimizer fit.',flush=True)


def train(root,arm):
 import torch,signal
 from compute_trainium.causal_associative_workspace import AssociativeWorkspace
 from compute_trainium.residual_matrix_work import MatrixWork
 from contextlib import nullcontext
 out=root/ID;p=json.loads((out/'plan.json').read_text());start=time.time();deadline=min(start+600,p['absolute_deadline_unix']);torch.set_num_threads(16);torch.manual_seed(p['seed']);report=dict(status='STOPPED_ASSOCIATIVE_ARM',assigned_chips=0);done=0;telemetry=[]
 def interrupted(number,frame):raise KeyboardInterrupt('Registered arm stopped')
 signal.signal(signal.SIGTERM,interrupted)
 try:
  maps=torch.load(root/p['source_map']['relative'],weights_only=True);raw=torch.load(out/'PRIVATE-raw-cache.pt',weights_only=True);h,y,d=load(root,p,'train',8192);m=AssociativeWorkspace(mode=arm);opt=torch.optim.AdamW(m.parameters(),lr=p['lr'],weight_decay=p['weight_decay']);rng=torch.Generator().manual_seed(p['query_seed']);warm=[]
  for step in range(p['updates']):
   guard(root,deadline);tick=time.time();ix=maps['fit'][step*4:step*4+4];q=torch.randperm(159,generator=rng)[:32].sort().values
   if not torch.equal(raw['positions'][ix],q.expand(4,-1)):raise AssertionError('Frozen natural-target schedule changed')
   opt.zero_grad(set_to_none=True);counter=MatrixWork() if step<2 else None
   with counter if counter else nullcontext():
    logp=m(h[ix],y[ix],q,y[ix][:,q],raw['logp'][ix]);loss=-logp.mean()
    if step==0 and not torch.equal(logp,raw['logp'][ix]):raise AssertionError('Initial ordinary identity failed')
    loss.backward()
   if counter:report.setdefault('matrix_operations_measured_updates',[]).append(counter.operations)
   norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();m.project_();done=step+1;seconds=time.time()-tick;warm.append(seconds);telemetry.append(dict(update=done,loss=float(loss.detach()),gradient_norm=float(norm),strength=float(m.strength.detach()),seconds=seconds))
   if step%16==0:(out/f'{arm}-progress.json').write_text(json.dumps(dict(update=done,wall_seconds=time.time()-start)))
   if step==15 and time.time()+sum(warm[1:])/15*(p['updates']-16)>deadline:raise RuntimeError('Measured fit exceeds registered allowance')
  torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=rng.get_state(),updates=done),out/f'PRIVATE-{arm}-state.pt');report.update(status='COMPLETE_ASSOCIATIVE_ARM',branch_parameters=sum(x.numel() for x in m.parameters()),warm_update_seconds=sum(warm[1:])/len(warm[1:]),label_exposures=done*128,temperature=None if arm=='bag' else float(m.log_temperature.detach().exp()),strength=float(m.strength.detach()))
 except BaseException as e:report.update(error_type=type(e).__name__,error=str(e));raise
 finally:
  report.update(updates=done,wall_seconds=time.time()-start);(out/f'{arm}-report.json').write_text(json.dumps(report,indent=2)+'\n');(out/f'{arm}-telemetry.json').write_text(json.dumps(telemetry))
  if done and report['status']!='COMPLETE_ASSOCIATIVE_ARM':torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=rng.get_state(),updates=done),out/f'PRIVATE-{arm}-interrupted-state.pt')


def raw_cache(root,p,out,h,y,head,maps):
 import torch
 from compute_trainium.stable_target_loss import stable_target_log_probs
 from compute_trainium.residual_matrix_work import MatrixWork
 start=time.time();deadline=min(start+480,p['absolute_deadline_unix']);positions=torch.zeros(8192,32,dtype=torch.long);logp=torch.full((8192,32),float('nan'),dtype=torch.float64);rng=torch.Generator().manual_seed(p['query_seed']);counter=None;operations=0
 with torch.no_grad():
  for stage,indices in [('fit',maps['fit']),('calibration',maps['calibration'])]:
   for step in range(0,len(indices),4):
    guard(root,deadline);ix=indices[step:step+4];q=torch.randperm(159,generator=rng)[:32].sort().values if stage=='fit' else torch.arange(127,159);positions[ix]=q;counter=MatrixWork()
    with counter:logp[ix]=stable_target_log_probs(h[ix][:,q]@head.T,y[ix][:,q]).double()
    operations+=counter.operations
    if step%64==0:(out/'raw-progress.json').write_text(json.dumps(dict(stage=stage,index=step,wall_seconds=time.time()-start)))
 torch.save(dict(positions=positions,logp=logp),out/'PRIVATE-raw-cache.pt');return dict(wall_seconds=time.time()-start,matrix_operations=operations,source_views=len(maps['fit'])+len(maps['calibration']),sha256=file_sha256(out/'PRIVATE-raw-cache.pt'))


def calibrate(p,h,y,d,maps,raw):
 import torch
 import numpy as np
 from torch.nn import functional as F
 from compute_trainium.causal_associative_workspace import prefix_weights,target_mass,mixture_log_probability
 sums=np.zeros((5,7,3));counts=np.zeros(3);q=torch.arange(127,159)
 with torch.no_grad():
  for i in maps['calibration'].tolist():
   similarities=F.normalize(h[i:i+1,q],dim=-1)@F.normalize(h[i:i+1],dim=-1).transpose(-1,-2);domain=int(d[i]);counts[domain]+=1
   for ti,temperature in enumerate(p['temperatures']):
    weights=prefix_weights(similarities*temperature,q);mass=target_mass(weights,y[i:i+1],q,y[i:i+1,q]);g=torch.tensor(p['mixture_weights'],dtype=torch.float64)[:,None,None];nll=-mixture_log_probability(raw['logp'][i:i+1],mass,g).mean((1,2));sums[ti,:,domain]+=nll.numpy()
 values=(sums/counts)@np.array([.6,.2,.2]);ti,wi=np.unravel_index(values.argmin(),values.shape);return dict(temperature=p['temperatures'][ti],weight=p['mixture_weights'][wi],nll=float(values[ti,wi]),raw_nll=float(values[0,0]),grid_nll=values.tolist(),domain_counts=counts.tolist(),selection_sources='TRAIN calibration groups only',boundary_temperature=ti in [0,4],boundary_weight=wi in [0,6])


def recover_check(path,mode,p,h,y,head,indices):
 import torch
 from compute_trainium.causal_associative_workspace import AssociativeWorkspace
 from compute_trainium.residual_data_volume import restore_branch
 from compute_trainium.stable_target_loss import stable_target_log_probs
 copies=[];sha=file_sha256(path)
 for _ in range(2):
  m=AssociativeWorkspace(mode=mode);opt=torch.optim.AdamW(m.parameters(),lr=p['lr'],weight_decay=p['weight_decay']);rng=torch.Generator();updates=restore_branch(path,m,opt,rng)
  if updates!=1536:raise AssertionError('Complete endpoint required')
  q=torch.randperm(159,generator=rng)[:32].sort().values;ix=indices[:4];raw=stable_target_log_probs(h[ix][:,q]@head.T,y[ix][:,q]).detach();loss=-m(h[ix],y[ix],q,y[ix][:,q],raw).mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();m.project_();copies.append((m.state_dict(),opt.state_dict(),rng.get_state(),float(loss.detach())))
 if copies[0][3]!=copies[1][3] or not torch.equal(copies[0][2],copies[1][2]):raise AssertionError('Replay loss/RNG changed')
 for k,v in copies[0][0].items():
  if not torch.equal(v,copies[1][0][k]):raise AssertionError('Replay model changed')
 for k,fields in copies[0][1]['state'].items():
  for f,v in fields.items():
   if isinstance(v,torch.Tensor) and not torch.equal(v,copies[1][1]['state'][k][f]):raise AssertionError('Replay moments changed')
 if file_sha256(path)!=sha:raise AssertionError('Scientific state changed')
 return dict(exact_restore_and_projected_update_replay=True,disposable_updates=2,checkpoint_unchanged=True)


def evaluate(root,p,out,head,selected):
 import torch
 from compute_trainium.causal_associative_workspace import AssociativeWorkspace,fixed_cache
 from compute_trainium.stable_target_loss import stable_target_log_probs
 from compute_trainium.residual_screen_analysis import contrast
 from compute_trainium.dependency_analysis import interval
 h,y,d=load(root,p,'tune',448);models={};versions=[]
 for arm in p['arm_modes']:
  m=AssociativeWorkspace(mode=arm).eval();m.load_state_dict(torch.load(out/f'PRIVATE-{arm}-state.pt',weights_only=True)['model']);m.requires_grad_(False);models[arm]=m;versions.append((m,tuple(v._version for v in m.parameters())))
 old=json.loads((root/p['bindings']['prior_one']['relative']).read_text());rows=[];tokens=[];q=torch.arange(127,159);max_error=0.;head_version=head._version
 with torch.no_grad():
  for i in range(448):
   guard(root,p['absolute_deadline_unix']);(out/'evaluation-progress.json').write_text(json.dumps(dict(index=i)));features=h[i:i+1];labels=y[i:i+1];target=labels[:,q];raw=stable_target_log_probs(features[:,q]@head.T,target).double();values=dict(raw=raw,fixed=fixed_cache(features,labels,q,target,raw,selected['temperature'],selected['weight']))
   for arm,m in models.items():values[arm]=m(features,labels,q,target,raw)
   values['rotated']=models['learned'](features,labels,q,target,raw,rotate=True);row=dict(index=i,domain=int(d[i]),group=p['evaluation_groups'][i],real_targets=32,old_one=old[i]['workspace_nll'],**{k:float(-v.mean()) for k,v in values.items()});err=abs(row['raw']-old[i]['raw_nll']);max_error=max(err,max_error)
   if (old[i]['index'],old[i]['domain'],old[i]['group'])!=(i,int(d[i]),p['evaluation_groups'][i]) or err>1e-5 or any(not math.isfinite(v) for k,v in row.items() if k!='group'):raise AssertionError('Raw parity,source identity or finite score failed')
   rows.append(row);repeat=((labels[:,None,:]==target[:,:,None])&(torch.arange(159)[None,None,:]<q[None,:,None])).any(-1)
   for kind in [False,True]:
    mask=repeat[0]==kind
    if mask.any():tokens.append(dict(index=i,domain=int(d[i]),group=p['evaluation_groups'][i],repeat=kind,real_targets=int(mask.sum()),**{k:float(-v[0,mask].mean()) for k,v in values.items()}))
 if head._version!=head_version or any(tuple(v._version for v in m.parameters())!=version for m,version in versions):raise AssertionError('Scoring altered weights')
 (out/'PRIVATE-scores.json').write_text(json.dumps(rows));(out/'PRIVATE-repeat-scores.json').write_text(json.dumps(tokens));fields=['raw','fixed','bag','cosine','learned','rotated','old_one'];scores={k:contrast(rows,[r[k] for r in rows]) for k in fields};contrasts={k+'_minus_learned':contrast(rows,[r[k]-r['learned'] for r in rows]) for k in ['raw','fixed','bag','cosine','rotated','old_one']};thresholds=dict(raw_minus_learned=.01,fixed_minus_learned=.005,bag_minus_learned=.005,cosine_minus_learned=.005,rotated_minus_learned=.005);gate=all(contrasts[k]['mean']>=bar and contrasts[k]['conditional_source_group_95_interval'][0]>0 for k,bar in thresholds.items());domains={name:{k:interval([r for r in rows if r['domain']==di],[r[k] for r in rows if r['domain']==di]) for k in fields} for di,name in enumerate(['web','math','code'])};strata={str(kind):{k:interval([r for r in tokens if r['repeat']==kind],[r[k] for r in tokens if r['repeat']==kind]) for k in fields if k!='old_one'} for kind in [False,True]}
 return dict(scores=scores,contrasts=contrasts,domains=domains,repeat_strata=strata,learned_association_gate=gate,thresholds=thresholds,max_raw_error=max_error,reserve_scored=False,scope='Reused development panel/one seed.Observed-prefix memory,not general reasoning,allocation,from-scratch or matched-compute efficiency.')


def run(root):
 import torch
 out=root/ID;path=out/'plan.json';p=json.loads(path.read_text());start=time.time();torch.set_num_threads(16);jobs={};logs=[];report=dict(status='STOPPED_CAUSAL_ASSOCIATIVE_SCREEN',assigned_chips=0,backbone_updates=0)
 try:
  for name,sha in p['source_sha256'].items():
   if file_sha256(name)!=sha:raise AssertionError('Frozen source changed')
  for name,sha in p['retained_feature_sha256'].items():
   if file_sha256(root/name)!=sha:raise AssertionError('Frozen features changed')
  for ref in p['bindings'].values():
   if file_sha256(root/ref['relative'])!=ref['sha256']:raise AssertionError('Frozen comparator changed')
  if file_sha256(root/p['checkpoint_relative'])!=p['checkpoint_sha256'] or file_sha256(root/p['source_map']['relative'])!=p['source_map']['sha256']:raise AssertionError('Frozen base/source mapping changed')
  maps=torch.load(root/p['source_map']['relative'],weights_only=True);h,y,d=load(root,p,'train',8192);head=torch.load(root/p['checkpoint_relative'],weights_only=True,mmap=True)['model']['lm_head.weight'].detach();report['raw_cache']=raw_cache(root,p,out,h,y,head,maps);write(out,'raw-cache-receipt.json',report['raw_cache']);fit_start=time.time()
  for arm in p['arm_modes']:
   log=(out/f'{arm}-train.log').open('x');logs.append(log);jobs[arm]=subprocess.Popen([sys.executable,'-m','scripts.causal_associative_screen','--root',str(root),'--arm',arm],stdout=log,stderr=subprocess.STDOUT)
  while any(job.poll() is None for job in jobs.values()):
   guard(root,min(fit_start+600,p['absolute_deadline_unix']))
   if any(job.poll() not in [None,0] for job in jobs.values()):raise RuntimeError('A registered fit stopped')
   time.sleep(.5)
  if any(job.returncode for job in jobs.values()):raise RuntimeError('An arm failed')
  report['arms']={arm:json.loads((out/f'{arm}-report.json').read_text()) for arm in jobs};audits={}
  for arm in jobs:
   saved_path=out/f'PRIVATE-{arm}-state.pt';saved=torch.load(saved_path,weights_only=True);audits[arm]=qualify(root,p,h,y,d,head,saved=saved,mode=arm);audits[arm].update(recover_check(saved_path,arm,p,h,y,head,maps['fit']))
  report['audits']=audits;write(out,'mature-audit.json',audits);raw=torch.load(out/'PRIVATE-raw-cache.pt',weights_only=True);report['calibration']=calibrate(p,h,y,d,maps,raw);write(out,'calibration.json',report['calibration']);report.update(evaluate(root,p,out,head,report['calibration']));report.update(status='COMPLETE_AUDITED_CAUSAL_ASSOCIATIVE_SCREEN',checkpoint_sha256={arm:file_sha256(out/f'PRIVATE-{arm}-state.pt') for arm in jobs})
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
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');p.add_argument('--arm');a=p.parse_args();train(a.root,a.arm) if a.arm else run(a.root) if a.run else prepare(a.root)
