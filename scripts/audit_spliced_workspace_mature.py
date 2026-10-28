"""Independent HF dense2D full-gradient oracle at the actual sparse endpoint."""
import argparse,json,os,time,shutil,resource
from pathlib import Path


def run(a):
 import torch
 from transformers import AutoModelForCausalLM
 from compute_trainium.execution_isolation import managed_running
 from compute_trainium.training import file_sha256
 from compute_trainium.workspace_cpu_reconstruction import load_streamed_model
 from compute_trainium.stable_target_loss import stable_target_log_probs
 if os.environ.get('USE_TORCH_XLA')!='0' or managed_running():raise RuntimeError('Closed Native owner and independent CPU environment required')
 p=json.loads(a.plan.read_text());discovery=a.arm is not None;count=p['updates'][a.arm] if discovery else 101
 workspace_mode=a.arm if discovery else p.get('workspace_mode','latent')
 native=a.root/'runs'/(p.get('arm_runs',{}).get(a.arm,p['identity']+'-'+a.arm) if discovery else p['warm_phase']);receipt=json.loads((native/'report.json').read_text())
 if receipt['status']!=('COMPLETE_SPARSE_DISCOVERY_ARM_PENDING_MATURE_HF_AUDIT' if discovery else 'COMPLETE_SPARSE_WARM_PENDING_MATURE_AUDIT'):raise AssertionError('Complete all64 trained endpoint states required')
 identity='spliced-workspace-discovery-audit-'+a.arm+'-v1' if discovery else p['identity']+'-mature-audit'
 out=a.root/identity;public=Path('results/spliced_workspace_discovery_20261028')/identity if discovery else Path('results/spliced_workspace_mature_20261028')/p['identity']
 if out.exists() or public.exists():raise FileExistsError('Retain every independent audit attempt')
 started=time.time();out.mkdir();public.mkdir(parents=True);torch.set_num_threads(12)
 def guard():
  available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
  if available<512*1024**3 or shutil.disk_usage(a.root).free<35*1024**3 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>64*1024**3 or time.time()-started>1200:raise RuntimeError('Mature CPU resource/time guard')
 shards=[native/f'rank{k}-u{count}.zip' for k in range(64)];gradients=[native/f'rank{k}-mature-gradients.pt' for k in range(64)]
 plan=dict(registered_unix=started,parent_plan_sha256=file_sha256(a.plan),native_report_sha256=file_sha256(native/'report.json'),source_sha256={n:file_sha256(n) for n in ['scripts/audit_spliced_workspace_mature.py','compute_trainium/workspace_cpu_reconstruction.py','compute_trainium/stable_target_loss.py']},checkpoint_sha256=[file_sha256(f) for f in shards],gradient_sha256=[file_sha256(f) for f in gradients],inputs_sha256=file_sha256(native/'PRIVATE-mature-view.pt'),optimizer_updates=0,assigned_chips=0,named_gradient_limit=.02,maximum_seconds=1200)
 for d in [out,public]:(d/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 result=dict(status='STOPPED_SPARSE_MATURE_CPU_AUDIT',optimizer_updates=0,assigned_chips=0)
 def progress(stage):guard();(out/'progress.json').write_text(json.dumps(dict(stage=stage,wall_seconds=time.time()-started)))
 try:
  progress('restore_all64_model_shards_into_independent_HF')
  model=AutoModelForCausalLM.from_pretrained(a.root/'workspace-olmo2-stage1-base-v1',local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').eval()
  load_streamed_model(shards,model,label='model');version=tuple(v._version for v in model.parameters())
  view=torch.load(native/'PRIVATE-mature-view.pt',weights_only=True);full=torch.cat([view['prefix'],view['continuation']],1)
  tokens=torch.cat([full[:,:-1],full.new_full((1,1),model.config.pad_token_id)],1);targets=full[:,1:];anchors=torch.tensor(p['mature_anchors']);keep=torch.ones(159,dtype=torch.bool);keep[anchors]=False;raw_positions=torch.arange(159)[keep]
  progress('independent_dense2D_forward_and_backward')
  raw=model.model(tokens,use_cache=False).last_hidden_state;z=raw[:,anchors];positions=torch.cat([torch.arange(160),anchors]);steps=torch.cat([torch.zeros(160),torch.ones(len(p['mature_anchors']))])
  allow=(positions[None,:]<=positions[:,None])&(steps[None,:]<=steps[:,None])&((steps[None,:]==0)|(positions[None,:]==positions[:,None]));mask=torch.zeros(160+len(p['mature_anchors']),160+len(p['mature_anchors'])).masked_fill(~allow,torch.finfo(torch.float32).min)[None,None]
  if workspace_mode=='ordinary':
   per_target=-stable_target_log_probs(model.lm_head(raw[:,:159]).double(),targets);raw_loss=per_target[:,raw_positions].mean();mixed_loss=per_target[:,anchors].mean();loss=per_target.mean()
  else:
   initial=model.model.norm(model.model.embed_tokens(torch.full_like(tokens[:,:len(p['mature_anchors'])],p['pause_token_id']))) if workspace_mode=='constant' else z
   refined=model.model(inputs_embeds=torch.cat([model.model.embed_tokens(tokens),initial],1),attention_mask=mask,position_ids=positions[None],use_cache=False).last_hidden_state[:,160:]
   raw_terms=-stable_target_log_probs(model.lm_head(raw[:,raw_positions]).double(),targets[:,raw_positions])
   mixed_terms=-stable_target_log_probs(model.lm_head(.5*z+.5*refined).double(),targets[:,anchors])
   raw_loss=raw_terms.mean();mixed_loss=mixed_terms.mean();loss=(raw_terms.sum()+mixed_terms.sum())/159
  loss.backward()
  progress('compare_every_named_gradient_to_all64_Native_shards');actual=[torch.load(f,weights_only=True,mmap=True) for f in gradients];errors={};difference=denominator=0.
  for name,v in model.named_parameters():
   if v.grad is None:raise AssertionError('All179 independent gradients required')
   diff=ref=0.
   for rank in range(64):
    wanted=v.grad.chunk(64,0)[rank].reshape(-1);have=actual[rank][name].reshape(-1)
    for k in range(0,wanted.numel(),65536):
     expected=wanted[k:k+65536].double();observed=have[k:k+65536].double();diff+=float((observed-expected).square().sum());ref+=float(expected.square().sum())
   errors[name]=(diff/max(ref,1e-30))**.5;difference+=diff;denominator+=ref
  result.update(parameters_checked=len(errors),largest_named_relative_l2=max(errors.values()),gradient_relative_l2=(difference/max(denominator,1e-30))**.5,worst_named_parameter=max(errors,key=errors.get),joint_loss=float(loss.detach()),raw_subset_nll=float(raw_loss.detach()),sparse_anchor_nll=float(mixed_loss.detach()),native_loss=receipt['mature_loss'])
  if max(errors.values())>.02:raise AssertionError('Mature sparse named gradient exceeds unchanged2% gate')
  if version!=tuple(v._version for v in model.parameters()):raise AssertionError('Audit changed trained model')
  result.update(status='PASS_SPLICED_INDEPENDENT_MATURE_CPU_AUDIT',arm=a.arm,endpoint_optimizer_counter=count,scope='Zero-update independent HF numerical audit at the saved uniform-NTP endpoint;all179 gradients checked against64 Native shards.Each natural label appears once,with independent per-position workspaces.No held-out quality,efficiency,allocation or random-init claim.')
 except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
 finally:
  result.update(wall_seconds=time.time()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,plan_sha256=file_sha256(out/'plan.json'))
  for d in [out,public]:(d/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--arm',choices=['ordinary','latent','constant']);run(p.parse_args())
