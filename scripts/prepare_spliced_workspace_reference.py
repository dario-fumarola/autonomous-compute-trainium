"""Independent full-size HF oracle for uniformly normalized spliced NTP."""
import argparse,json,time,os,shutil,resource
from pathlib import Path


def run(root,mode='latent'):
 import torch
 from transformers import AutoModelForCausalLM
 from compute_trainium.execution_isolation import managed_running
 from compute_trainium.training import file_sha256
 from compute_trainium.workspace_olmo_candidate import OlmoWorkspaceConfig,rotary_tensors
 from compute_trainium.sparse_workspace_candidate import IndependentSparseWorkspaceCandidate
 from compute_trainium.sparse_workspace_training import pack_view
 from compute_trainium.spliced_workspace_training import partition_positions,spliced_objective
 from compute_trainium.workspace_view_batches import WorkspaceViewBatches
 from compute_trainium.stable_target_loss import stable_target_log_probs
 if os.environ.get('USE_TORCH_XLA')!='0' or managed_running():raise RuntimeError('Closed Native owner and independent CPU environment required')
 identity='spliced-workspace-constant-cpu-reference-v1' if mode=='constant' else 'spliced-workspace-cpu-reference-v1'
 out=root/identity;public=Path('results/spliced_workspace_reference_20261028')/('constant-v1' if mode=='constant' else '')
 if out.exists() or public.exists():raise FileExistsError('Preserve each reference attempt')
 if shutil.disk_usage(root).free<42*1024**3:raise RuntimeError('Full gradient plus35GiB floor required')
 out.mkdir();public.mkdir(parents=True);started=time.time();torch.set_num_threads(12)
 parent=json.loads((root/'sparse-workspace-native-v1/plan.json').read_text());cfg=OlmoWorkspaceConfig(**parent['model_config'])
 def guard():
  available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
  if available<512*1024**3 or shutil.disk_usage(root).free<35*1024**3 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>64*1024**3 or time.time()-started>900:raise RuntimeError('Protected CPU reference allowance')
 plan=dict(registered_unix=started,model_config=parent['model_config'],checkpoint_relative=parent['checkpoint_relative'],checkpoint_sha256=parent['checkpoint_sha256'],source_index=0,data_relative='workspace-view-batches-v1',positions=160,query_count=32,anchors=list(range(31,159,4)),mix=.5,mode='latent',objective='Every authentic target once;raw127+mixed32 CE sums/159',independent_queries=True,source_sha256={n:file_sha256(n) for n in ['scripts/prepare_spliced_workspace_reference.py','compute_trainium/sparse_workspace_candidate.py','compute_trainium/spliced_workspace_training.py','compute_trainium/stable_target_loss.py']},seconds_limit=900,cpu_allowance_gib=64,named_gradient_limit=.02,optimizer_updates=0)
 plan['mode']=mode;plan['pause_token_id']=json.loads((root/'workspace-four-arm-discovery-v3/plan.json').read_text())['pause_token_id']
 for d in [out,public]:(d/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 result=dict(status='STOPPED_SPLICED_CPU_REFERENCE',assigned_chips=0,optimizer_updates=0)
 try:
  view=WorkspaceViewBatches(root/plan['data_relative'],'train').at(0);tokens,targets=pack_view(view,cfg.pad_token_id);anchors,where=partition_positions(torch.tensor(plan['anchors']));cos,sin=rotary_tensors(cfg,160)
  state=torch.load(root/parent['checkpoint_relative'],weights_only=True,mmap=True)['model']
  candidate=IndependentSparseWorkspaceCandidate(cfg,activation_checkpointing=True).eval();candidate.load_state_dict(state)
  reference=AutoModelForCausalLM.from_pretrained(root/'workspace-olmo2-stage1-base-v1',local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').eval();reference.load_state_dict(state);del state
  actual=candidate.sparse_forward(tokens,cos,sin,anchors,raw_positions=where,mode=mode,pause_id=plan['pause_token_id'])
  loss=(-stable_target_log_probs(actual['raw_logits'].double(),targets[:,where]).sum()-stable_target_log_probs(actual['mixed_logits'].double(),targets[:,anchors]).sum())/159
  loss.backward();guard()
  raw=reference.model(tokens,use_cache=False).last_hidden_state;z=raw[:,anchors];positions=torch.cat([torch.arange(160),anchors]);steps=torch.cat([torch.zeros(160),torch.ones(32)])
  allowed=(positions[None,:]<=positions[:,None])&(steps[None,:]<=steps[:,None])&((steps[None,:]==0)|(positions[None,:]==positions[:,None]))
  mask=torch.zeros(192,192).masked_fill(~allowed,torch.finfo(torch.float32).min)[None,None]
  initial=reference.model.norm(reference.model.embed_tokens(torch.full_like(tokens[:,:32],plan['pause_token_id']))) if mode=='constant' else z
  refined=reference.model(inputs_embeds=torch.cat([reference.model.embed_tokens(tokens),initial],1),attention_mask=mask,position_ids=positions[None],use_cache=False).last_hidden_state[:,160:]
  raw_logits=reference.lm_head(raw[:,where]);mixed_logits=reference.lm_head(.5*z+.5*refined)
  expected=(-stable_target_log_probs(raw_logits.double(),targets[:,where]).sum()-stable_target_log_probs(mixed_logits.double(),targets[:,anchors]).sum())/159
  expected.backward();guard();errors={};difference=denominator=0.
  for (name,p),(other,q) in zip(candidate.named_parameters(),reference.named_parameters(),strict=True):
   if name!=other or p.grad is None or q.grad is None:raise AssertionError('Identical complete inventory required')
   diff=ref=0.;x=p.grad.reshape(-1);y=q.grad.reshape(-1)
   for k in range(0,x.numel(),65536):
    a=x[k:k+65536].double();b=y[k:k+65536].double();diff+=float((a-b).square().sum());ref+=float(b.square().sum())
   errors[name]=(diff/max(ref,1e-30))**.5;difference+=diff;denominator+=ref
  result.update(parameters_checked=len(errors),largest_named_relative_l2=max(errors.values()),gradient_relative_l2=(difference/max(denominator,1e-30))**.5,worst_named_parameter=max(errors,key=errors.get),raw_logits_max_abs_error=float((actual['raw_logits'].detach()-raw_logits.detach()).abs().max()),mixed_logits_max_abs_error=float((actual['mixed_logits'].detach()-mixed_logits.detach()).abs().max()),cached_loss=float(loss.detach()),independent_hf_loss=float(expected.detach()))
  if max(errors.values())>.02:raise AssertionError('Full-size spliced gradient exceeds unchanged2% gate')
  path=out/'independent-gradients.pt';torch.save({n:p.grad for n,p in reference.named_parameters()},path);torch.save(dict(tokens=tokens,targets=targets,anchors=anchors,raw_positions=where),out/'PRIVATE-inputs.pt')
  result.update(status='PASS_SPLICED_FULL_SIZE_CPU_REFERENCE',gradient_relative=str(path.relative_to(root)),gradient_sha256=file_sha256(path),scope='One authentic TRAIN view at actual C0 weights.Each natural target receives one CE;independent per-position workspace queries.No fitting,held-out readout,reasoning or efficiency claim.')
 except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
 finally:
  result.update(wall_seconds=time.time()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,plan_sha256=file_sha256(out/'plan.json'))
  for d in [out,public]:(d/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',choices=['latent','constant'],default='latent');a=p.parse_args();run(a.root,a.mode)
