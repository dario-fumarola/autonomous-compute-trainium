"""Small frozen-base residual screen;all learned parameters are in the branch."""
import argparse,json,time,os,resource,shutil
from pathlib import Path


def run(a):
 import torch
 from compute_trainium.residual_workspace import ResidualWorkspace
 from compute_trainium.stable_target_loss import stable_target_log_probs
 from compute_trainium.training import file_sha256
 from compute_trainium.residual_matrix_work import MatrixWork
 from compute_trainium.residual_data_volume import source_indices,restore_branch
 from contextlib import nullcontext
 p=json.loads(a.plan.read_text());out=a.root/p['identity'];start=time.time();torch.set_num_threads(16);torch.manual_seed(p['seed']);report=dict(status='STOPPED_RESIDUAL_SCREEN_ARM',arm=a.arm,backbone_updates=0,assigned_chips=0);completed=0;telemetry=[]
 import signal
 def interrupted(number,frame):raise KeyboardInterrupt('Registered screen interrupted')
 signal.signal(signal.SIGTERM,interrupted)
 if file_sha256(a.plan)!=a.plan_sha:raise AssertionError('Frozen screen required')
 if a.arm not in p['arm_modes']:raise AssertionError('Unregistered arm')
 def guard(stage,index):
  available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
  if available<512*1024**3 or shutil.disk_usage(a.root).free<35*1024**3 or os.statvfs(a.root).f_favail<10000 or time.time()>p['absolute_deadline_unix'] or time.time()-start>p['cpu_phase_seconds_limit']:raise RuntimeError('Frozen CPU fitting resource/time guard')
  (out/f'{a.arm}-progress.json').write_text(json.dumps(dict(stage=stage,index=index,wall_seconds=time.time()-start)))
 def load(split,count):
  h=torch.empty(count,159,p['model_config']['hidden_size']);y=torch.empty(count,159,dtype=torch.long);d=torch.empty(count,dtype=torch.long);seen=[]
  phases=p.get('feature_segments',{}).get(split,[p['feature_phase']])
  for phase in phases:
   for rank in range(64):
    v=torch.load(a.root/'runs'/phase/f'PRIVATE-rank{rank}-{split}.pt',weights_only=True,mmap=True);keep=v['indices']<count;indices=v['indices'][keep]
    if not len(indices):continue
    h[indices]=v['features'][keep];y[indices]=v['targets'][keep];d[indices]=v['domains'][keep];seen+=indices.tolist()
  if sorted(seen)!=list(range(count)):raise AssertionError('Exact frozen feature coverage required')
  return h,y,d
 try:
  unique_views=p.get('arm_unique_views',{}).get(a.arm,p['train_views']);mode=p.get('branch_modes',{}).get(a.arm,a.arm)
  guard('load_frozen_features',0);train,labels,domains=load('train',unique_views);test,truth,td=load('tune',p['evaluation_anchors']);state=torch.load(a.root/p['checkpoint_relative'],weights_only=True,mmap=True)['model'];head=state['lm_head.weight'].detach();del state;head_version=head._version
  model=ResidualWorkspace(width=p['width'],mode=mode);optimizer=torch.optim.AdamW(model.parameters(),lr=p['lr'],weight_decay=p['weight_decay']);rng=torch.Generator().manual_seed(p['seed']+1);telemetry=[];q=p['queries'];batch=p['batch_views'];warm=[];initial_update=0
  if p.get('resume_states'):
   resume=p['resume_states'][a.arm];path=a.root/resume['relative']
   if file_sha256(path)!=resume['sha256']:raise AssertionError('Frozen branch prefix changed')
   initial_update=restore_branch(path,model,optimizer,rng)
   if initial_update!=resume['updates']:raise AssertionError('Exact registered prefix counter required')
   report.update(retained_prefix_updates=initial_update,exact_prefix_restore=True)
  model.train()
  for update in range(initial_update,p['updates']):
   guard('train',update);tick=time.time();indices=source_indices(update,batch,unique_views);anchors=torch.randperm(159,generator=rng)[:q].sort().values;h=train[indices];target=labels[indices][:,anchors];optimizer.zero_grad(set_to_none=True);counter=MatrixWork() if update<initial_update+2 else None
   with counter if counter is not None else nullcontext():
    adjusted=model(h,anchors);logits=adjusted@head.T;loss=-stable_target_log_probs(logits,target).mean()
    if update==0 and not torch.equal(adjusted,h[:,anchors]):raise AssertionError('Initial prediction must be exactly ordinary')
    loss.backward()
   if counter is not None:report.setdefault('matrix_operations_measured_updates',[]).append(counter.operations)
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();completed=update+1;seconds=time.time()-tick;telemetry.append(dict(update=update+1,loss=float(loss.detach()),gradient_norm=float(norm),seconds=seconds));warm.append(seconds)
   if update%16==0:(out/f'{a.arm}-telemetry.json').write_text(json.dumps(telemetry))
   # Measure actual path before retaining more than16 fitted updates.
   if update==initial_update+15 and time.time()+sum(warm[1:])/15*(p['updates']-initial_update-16+40)>min(p['absolute_deadline_unix'],start+1800):raise RuntimeError('Measured fit plus evaluation does not fit frozen cap')
  torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.get_state(),updates=p['updates']),out/f'PRIVATE-{a.arm}-state.pt');(out/f'{a.arm}-telemetry.json').write_text(json.dumps(telemetry))
  model.eval();rows=[];anchors=torch.arange(127,159)
  with torch.no_grad():
   for i in range(len(test)):
    guard('evaluate',i);h=test[i:i+1];targets=truth[i:i+1,127:159];raw=-stable_target_log_probs(h[:,127:159]@head.T,targets);work=-stable_target_log_probs(model(h,anchors)@head.T,targets)
    rows.append(dict(index=i,domain=int(td[i]),group=p['evaluation_groups'][i],raw_nll=float(raw.mean()),workspace_nll=float(work.mean()),first_raw_nll=float(raw[0,0]),first_workspace_nll=float(work[0,0])))
  if head._version!=head_version or train.requires_grad or test.requires_grad:raise AssertionError('Frozen predictor was changed')
  (out/f'PRIVATE-{a.arm}-scores.json').write_text(json.dumps(rows));report.update(status='COMPLETE_RESIDUAL_SCREEN_ARM',updates=p['updates'],new_updates=p['updates']-initial_update,branch_parameters=sum(x.numel() for x in model.parameters()),unique_train_views=unique_views,train_source_view_exposures=p['updates']*batch,label_exposures=p['updates']*batch*q,mean_warm_update_seconds=sum(warm[1:])/len(warm[1:]),parameters_frozen=True)
 except BaseException as error:report.update(error_type=type(error).__name__,error=str(error));raise
 finally:
  report.update(completed_updates=completed,wall_seconds=time.time()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);(out/f'{a.arm}-report.json').write_text(json.dumps(report,indent=2)+'\n');(out/f'{a.arm}-telemetry.json').write_text(json.dumps(telemetry))
  if completed and report['status']!='COMPLETE_RESIDUAL_SCREEN_ARM':torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.get_state(),updates=completed),out/f'PRIVATE-{a.arm}-interrupted-state.pt')


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--arm',required=True);run(p.parse_args())
