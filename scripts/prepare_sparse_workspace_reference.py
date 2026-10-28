"""Actual1.485B cached-vs-independent-HF dense2D numerical reference."""
import argparse,json,time,os,shutil,resource
from pathlib import Path


def run(root):
 import torch
 from transformers import AutoModelForCausalLM
 from compute_trainium.execution_isolation import managed_running
 from compute_trainium.training import file_sha256
 from compute_trainium.workspace_olmo_candidate import OlmoWorkspaceConfig,rotary_tensors
 from compute_trainium.sparse_workspace_candidate import SparseWorkspaceCandidate
 from compute_trainium.workspace_view_batches import WorkspaceViewBatches
 from compute_trainium.stable_target_loss import stable_target_log_probs
 if os.environ.get('USE_TORCH_XLA')!='0' or managed_running():raise RuntimeError('Closed Native owner and independent CPU environment required')
 out=root/'sparse-workspace-cpu-reference-v1';public=Path('results/sparse_workspace_reference_20261028')
 if out.exists() or public.exists():raise FileExistsError('Preserve every reference attempt')
 if shutil.disk_usage(root).free<42*1024**3:raise RuntimeError('Complete gradient plus35GiB disk floor required')
 out.mkdir();public.mkdir();started=time.time();torch.set_num_threads(12)
 parent=json.loads((root/'workspace-four-arm-discovery-v3/plan.json').read_text());cfg=OlmoWorkspaceConfig(**parent['model_config'])
 def guard():
  available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
  if available<512*1024**3 or shutil.disk_usage(root).free<35*1024**3 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>64*1024**3 or time.time()-started>900:raise RuntimeError('CPU reference protected resource/time allowance')
 plan=dict(registered_unix=started,model_config=parent['model_config'],checkpoint_relative=parent['checkpoint_relative'],checkpoint_sha256=parent['checkpoint_sha256'],data_relative=parent['data_relative'],source_index=0,positions=160,anchors=[127,135,143,151],mix=.5,mode='latent',loss='0.5 mean raw159 CE +0.5 mean mixed4 CE',source_sha256={n:file_sha256(n) for n in ['scripts/prepare_sparse_workspace_reference.py','compute_trainium/sparse_workspace_candidate.py','compute_trainium/stable_target_loss.py']},worker_memory_gib=0,cpu_allowance_gib=64,seconds_limit=900,optimizer_updates=0,named_gradient_limit=.02)
 for d in [out,public]:(d/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 result=dict(status='STOPPED_ACTUAL_SPARSE_CPU_REFERENCE',optimizer_updates=0,assigned_chips=0)
 try:
  view=WorkspaceViewBatches(root/parent['data_relative'],'train').at(0);full=torch.cat([view['prefix'],view['continuation']],1)
  tokens=torch.cat([full[:,:-1],full.new_full((1,1),cfg.pad_token_id)],1);targets=full[:,1:];anchors=torch.tensor(plan['anchors']);where=torch.arange(159);cos,sin=rotary_tensors(cfg,160)
  state=torch.load(root/parent['checkpoint_relative'],weights_only=True,mmap=True)['model']
  reference=AutoModelForCausalLM.from_pretrained(root/'workspace-olmo2-stage1-base-v1',local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').eval();reference.load_state_dict(state)
  candidate=SparseWorkspaceCandidate(cfg,activation_checkpointing=True).eval();candidate.load_state_dict(state);del state;guard()
  actual=candidate.sparse_forward(tokens,cos,sin,anchors,raw_positions=where)
  actual_loss=.5*(-stable_target_log_probs(actual['raw_logits'].double(),targets).mean())+.5*(-stable_target_log_probs(actual['mixed_logits'].double(),full[:,anchors+1]).mean())
  actual_loss.backward();guard()
  raw=reference.model(tokens,use_cache=False).last_hidden_state;z=raw[:,anchors];embedding=reference.model.embed_tokens(tokens)
  positions=torch.cat([torch.arange(160),anchors]);steps=torch.cat([torch.zeros(160),torch.ones(4)])
  allow=(positions[None,:]<=positions[:,None])&(steps[None,:]<=steps[:,None])
  mask=torch.zeros(164,164).masked_fill(~allow,torch.finfo(torch.float32).min)[None,None]
  refined=reference.model(inputs_embeds=torch.cat([embedding,z],1),attention_mask=mask,position_ids=positions[None],use_cache=False).last_hidden_state[:,160:]
  raw_logits=reference.lm_head(raw[:,:159]);mixed_logits=reference.lm_head(.5*z+.5*refined)
  expected_loss=.5*(-stable_target_log_probs(raw_logits.double(),targets).mean())+.5*(-stable_target_log_probs(mixed_logits.double(),full[:,anchors+1]).mean())
  expected_loss.backward();guard();errors={};difference=denominator=0.
  for (name,p),(other,q) in zip(candidate.named_parameters(),reference.named_parameters(),strict=True):
   if name!=other or p.grad is None or q.grad is None:raise AssertionError('Complete identical gradient inventory required')
   diff=ref=0.;a=p.grad.reshape(-1);b=q.grad.reshape(-1)
   for k in range(0,a.numel(),65536):
    aa=a[k:k+65536].double();bb=b[k:k+65536].double();diff+=float((aa-bb).square().sum());ref+=float(bb.square().sum())
   errors[name]=(diff/max(ref,1e-30))**.5;difference+=diff;denominator+=ref
  result.update(parameters_checked=len(errors),largest_named_relative_l2=max(errors.values()),gradient_relative_l2=(difference/max(denominator,1e-30))**.5,worst_named_parameter=max(errors,key=errors.get),raw_logits_max_abs_error=float((actual['raw_logits'].detach()-raw_logits.detach()).abs().max()),mixed_logits_max_abs_error=float((actual['mixed_logits'].detach()-mixed_logits.detach()).abs().max()),cached_loss=float(actual_loss.detach()),dense_hf_loss=float(expected_loss.detach()))
  if max(errors.values())>.02:raise AssertionError('Independent full-model gradient exceeds unchanged2% rule')
  path=out/'independent-gradients.pt';torch.save({n:p.grad for n,p in reference.named_parameters()},path)
  torch.save(dict(tokens=tokens,targets=targets,anchors=anchors,raw_positions=where),out/'PRIVATE-inputs.pt')
  result.update(status='PASS_ACTUAL_SPARSE_CPU_REFERENCE',gradient_relative=str(path.relative_to(root)),gradient_sha256=file_sha256(path),scope='Actual pretrainedC0 weights on one authentic TRAIN view.Zero updates.Cached differentiable prefixKV compared with independent HF eager dense2D mask,FP32 body/FP64 loss.No quality,efficiency or from-scratch claim.')
 except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
 finally:
  result.update(wall_seconds=time.time()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,plan_sha256=file_sha256(out/'plan.json'))
  for d in [out,public]:(d/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
