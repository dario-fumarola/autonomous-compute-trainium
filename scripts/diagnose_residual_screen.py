"""Post-fit numerical/recovery and TRAIN-only learning diagnosis;zero retained updates."""
import argparse,json,time,os
from pathlib import Path


def run(root,volume=False):
 import torch
 from compute_trainium.residual_workspace import ResidualWorkspace
 from compute_trainium.stable_target_loss import stable_target_log_probs
 from compute_trainium.training import file_sha256
 identity='residual-data-volume-v1' if volume else 'residual-workspace-screen-v1';out=root/identity;p=json.loads((out/'plan.json').read_text());public=Path('results/residual_data_volume_20261028' if volume else 'results/residual_workspace_screen_20261028');start=time.time();torch.set_num_threads(16)
 if json.loads((out/'report.json').read_text())['status']!='COMPLETE_FROZEN_RESIDUAL_SCREEN':raise AssertionError('Completed fixed endpoint required')
 registration=dict(registered_unix=start,deadline_unix=min(start+600,p['absolute_deadline_unix']),train_views=512,optimizer_updates_to_saved_models=0,development_rescoring=False,source_sha256=file_sha256(Path(__file__)),scope='Post-fit explanatory diagnosis,not a new scientific endpoint or promotion rule.First512 TRAIN views by frozen order;compare trained labels and final32 readout independently.Numerical checks use disposable copies;scientific checkpoints unchanged.')
 (public/'diagnostic-plan.json').write_text(json.dumps(registration,indent=2)+'\n')
 h=torch.empty(512,159,2048);y=torch.empty(512,159,dtype=torch.long);domain=torch.empty(512,dtype=torch.long)
 for rank in range(64):
  phase=p.get('feature_segments',{}).get('train',[p['feature_phase']])[0];v=torch.load(root/'runs'/phase/f'PRIVATE-rank{rank}-train.pt',weights_only=True,mmap=True);keep=v['indices']<512;ix=v['indices'][keep];h[ix]=v['features'][keep];y[ix]=v['targets'][keep];domain[ix]=v['domains'][keep]
 state=torch.load(root/p['checkpoint_relative'],weights_only=True,mmap=True)['model'];head=state['lm_head.weight'].detach();del state;result=dict(arms={},retained_optimizer_updates=0,assigned_chips=0)
 def check():
  available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
  if available<512*1024**3 or time.time()>registration['deadline_unix']:raise RuntimeError('Original reserve/finite diagnostic cap')
 for arm in p['arm_modes']:
  check();path=out/f'PRIVATE-{arm}-state.pt';before=file_sha256(path);saved=torch.load(path,weights_only=True);mode=p.get('branch_modes',{}).get(arm,arm);m=ResidualWorkspace(width=p['width'],mode=mode);m.load_state_dict(saved['model']);opt=torch.optim.AdamW(m.parameters(),lr=p['lr'],weight_decay=p['weight_decay']);opt.load_state_dict(saved['optimizer']);g=torch.Generator();g.set_state(saved['rng'])
  if not torch.equal(g.get_state(),saved['rng']) or any(not torch.equal(v,m.state_dict()[k]) for k,v in saved['model'].items()):raise AssertionError('Exact branch/RNG restore failed')
  for key,v in saved['optimizer']['state'].items():
   for field,value in v.items():
    if isinstance(value,torch.Tensor) and not torch.equal(value,opt.state_dict()['state'][key][field]):raise AssertionError('Exact optimizer moment restore failed')
  double=ResidualWorkspace(width=p['width'],mode=mode).double();double.load_state_dict(saved['model']);anchors=torch.arange(127,159)
  for model,features,projection in [(m,h[:1],head),(double,h[:1].double(),head.double())]:
   loss=-stable_target_log_probs(model(features,anchors)@projection.T,y[:1,127:159]).mean();loss.backward()
  errors={name:float((v.grad.double()-dict(double.named_parameters())[name].grad).norm()/dict(double.named_parameters())[name].grad.norm().clamp_min(1e-30)) for name,v in m.named_parameters()}
  if max(errors.values())>.02:raise AssertionError('Actual-shape branch gradient exceeds2% against FP64')
  del double;opt.zero_grad(set_to_none=True);m.eval();rng=torch.Generator().manual_seed(p['seed']+1);sums={name:[0.,0.,0.] for name in ['seen_raw','seen_work','tail_raw','tail_work']};counts=[0,0,0]
  with torch.no_grad():
   for step in range(128):
    check();indices=torch.arange(step*4,(step+1)*4);chosen=torch.randperm(159,generator=rng)[:32].sort().values
    for view,positions in [('seen',chosen),('tail',anchors)]:
     raw=-stable_target_log_probs(h[indices][:,positions]@head.T,y[indices][:,positions]).mean(-1);work=-stable_target_log_probs(m(h[indices],positions)@head.T,y[indices][:,positions]).mean(-1)
     for i,d in enumerate(domain[indices].tolist()):sums[view+'_raw'][d]+=float(raw[i]);sums[view+'_work'][d]+=float(work[i])
    for d in domain[indices].tolist():counts[d]+=1
  scores={key:sum(w*x/n for w,x,n in zip([.6,.2,.2],values,counts,strict=True)) for key,values in sums.items()}
  if file_sha256(path)!=before:raise AssertionError('Diagnostic changed saved endpoint')
  result['arms'][arm]=dict(exact_parameter_optimizer_rng_restore=True,named_gradient_errors=errors,max_named_gradient_relative_l2=max(errors.values()),train_scores=scores,trained_label_utility=scores['seen_raw']-scores['seen_work'],train_tail32_utility=scores['tail_raw']-scores['tail_work'],checkpoint_unchanged=True)
 result.update(status='COMPLETE_RESIDUAL_TRAIN_DIAGNOSIS_AND_CPU_AUDIT',wall_seconds=time.time()-start)
 for folder in [out,public]:(folder/'train-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--volume',action='store_true');a=p.parse_args();run(a.root,a.volume)
