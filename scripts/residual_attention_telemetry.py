"""TRAIN-only inspection of frozen attention and residual state geometry."""
import argparse,json,time,os
from pathlib import Path
from compute_trainium.training import file_sha256
ID='residual-attention-telemetry-v1';PUBLIC=Path('results/residual_attention_telemetry_20261028')


def prepare(root):
 from compute_trainium.execution_isolation import managed_running
 if managed_running():raise RuntimeError('No active Native owner')
 out=root/ID;out.mkdir(mode=0o700);PUBLIC.mkdir();parent=json.loads((root/'residual-data-volume-v1/plan.json').read_text());path=root/'residual-data-volume-v1/PRIVATE-fresh_one-state.pt';start=time.time()
 p=dict(identity=ID,registered_unix=start,absolute_deadline_unix=start+600,optimizer_updates=0,assigned_chips=0,train_views=512,anchors=list(range(127,159)),seed=7285,checkpoint_relative=str(path.relative_to(root)),checkpoint_sha256=file_sha256(path),feature_sha256={k:v for k,v in parent['retained_feature_sha256'].items() if k.endswith('-train.pt')},source_sha256={name:file_sha256(name) for name in ['scripts/residual_attention_telemetry.py','compute_trainium/residual_workspace.py','scripts/residual_memory_attribution.py']},scope='Post-attribution,TRAIN-only geometry diagnostic.First512 original TRAIN feature views and all32 tail queries.Fixed trained one-step vs its original random initialization.No vocabulary projection,loss,targets,held-out rescoring or model update.Compute attention entropy/effective support,self/top1 weights,value-mean energy,weighted-vs-uniform context and final-correction differences.Report all three source domains and unchanged parameter versions.The initialized output is zero by construction;zero initial output differences are not evidence of useful computation.No new promotion threshold or fit.')
 for folder in [out,PUBLIC]:(folder/'plan.json').write_text(json.dumps(p,indent=2)+'\n')


def run(root):
 import torch
 from torch.nn import functional as F
 from compute_trainium.residual_workspace import ResidualWorkspace,unit
 from scripts.residual_memory_attribution import guard
 out=root/ID;p=json.loads((out/'plan.json').read_text());start=time.time();torch.set_num_threads(16);report=dict(status='STOPPED_ATTENTION_TELEMETRY',optimizer_updates=0,assigned_chips=0,model_scores_computed=0)
 try:
  for name,sha in p['source_sha256'].items():
   if file_sha256(name)!=sha:raise AssertionError('Frozen telemetry source changed')
  path=root/p['checkpoint_relative']
  if file_sha256(path)!=p['checkpoint_sha256']:raise AssertionError('Trained state changed')
  h=torch.empty(512,159,2048);domains=torch.empty(512,dtype=torch.long);seen=[]
  for name,sha in p['feature_sha256'].items():
   if file_sha256(root/name)!=sha:raise AssertionError('Frozen TRAIN cache changed')
   data=torch.load(root/name,weights_only=True,mmap=True);keep=data['indices']<512;ix=data['indices'][keep];h[ix]=data['features'][keep];domains[ix]=data['domains'][keep];seen+=ix.tolist()
  if sorted(seen)!=list(range(512)):raise AssertionError('Exact TRAIN-only feature panel required')
  torch.manual_seed(p['seed']);initial=ResidualWorkspace(mode='one').eval().requires_grad_(False);trained=ResidualWorkspace(mode='one').eval().requires_grad_(False);trained.load_state_dict(torch.load(path,weights_only=True)['model']);a=torch.tensor(p['anchors']);mask=torch.arange(159)[None,:]<=a[:,None];count=(a+1).double();result={};max_error=0.
  with torch.no_grad():
   for label,m in [('initial',initial),('trained',trained)]:
    versions=tuple(v._version for v in m.parameters());rows=[]
    for i in range(512):
     guard(root,p['absolute_deadline_unix']);x=h[i:i+1];memory=m.input(unit(x));state=memory[:,a];cell=m.cell;q=cell.q(unit(state));k=cell.k(unit(memory));v=cell.v(unit(memory));scores=(q@k.transpose(-1,-2))*(128**-.5);w=scores.masked_fill(~mask,float('-inf')).softmax(-1);read=w@v;mean=v.cumsum(1)[:,a]/(a+1)[None,:,None];att=cell.o(read);after=state+att;ff=cell.down(F.silu(cell.up(unit(after))));delta=m.output(unit(after+ff));u=state+cell.o(mean);u=u+cell.down(F.silu(cell.up(unit(u))));uniform_delta=m.output(unit(u));err=float((x[:,a]+delta-m(x,a)).abs().max());max_error=max(max_error,err)
     if err>1e-5:raise AssertionError('Telemetry changed actual branch forward')
     entropy=-(w.double()*w.double().clamp_min(1e-300).log()).sum(-1);ratio=lambda u,v:torch.linalg.vector_norm(u,dim=-1)/torch.linalg.vector_norm(v,dim=-1).clamp_min(1e-12);mean_energy=mean.square().sum(-1)/(v.square().sum(-1).cumsum(1)[:,a]/(a+1)[None,:]).clamp_min(1e-12)
     metrics=dict(entropy_fraction_of_uniform=entropy/count.log(),effective_support_fraction=entropy.exp()/count,kl_to_uniform_nats=count.log()-entropy,largest_attention_weight=w.max(-1).values,self_attention_mass=w.gather(-1,a[None,:,None]).squeeze(-1),value_mean_energy_fraction=mean_energy,read_minus_uniform_relative_l2=ratio(read-mean,read),attention_update_relative_l2=ratio(att,state),ffn_update_relative_l2=ratio(ff,state),correction_relative_l2=ratio(delta,x[:,a]),correction_minus_uniform_relative_l2=ratio(delta-uniform_delta,delta))
     rows.append(dict(domain=int(domains[i]),**{k:float(value.mean()) for k,value in metrics.items()}))
    if tuple(v._version for v in m.parameters())!=versions:raise AssertionError('Telemetry changed parameters')
    result[label]={name:{key:sum(r[key] for r in rows if r['domain']==d)/sum(r['domain']==d for r in rows) for key in rows[0] if key!='domain'} for d,name in enumerate(['web','math','code'])}
    result[label]['weighted']={key:sum(weight*result[label][name][key] for name,weight in [('web',.6),('math',.2),('code',.2)]) for key in rows[0] if key!='domain'}
  movement={name:dict(initial_l2=float(v.norm()),trained_minus_initial_l2=float((dict(trained.named_parameters())[name]-v).norm())) for name,v in initial.named_parameters()}
  if file_sha256(path)!=p['checkpoint_sha256']:raise AssertionError('Telemetry changed checkpoint')
  report.update(status='COMPLETE_TRAIN_ONLY_ATTENTION_TELEMETRY',metrics=result,parameter_movement=movement,max_branch_identity_error=max_error,checkpoint_unchanged=True,scope=p['scope'])
 except BaseException as e:report.update(error_type=type(e).__name__,error=str(e));raise
 finally:
  report.update(wall_seconds=time.time()-start,plan_sha256=file_sha256(out/'plan.json'))
  for folder in [out,PUBLIC]:(folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args();run(a.root) if a.run else prepare(a.root)
