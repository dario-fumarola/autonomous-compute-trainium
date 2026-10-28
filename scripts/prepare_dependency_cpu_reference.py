"""Independent full-size1024-position HF forward oracle;no model fitting."""
import argparse,json,os,time,resource,shutil
from pathlib import Path


def run(root):
 import torch
 import torch.nn.functional as F
 from transformers import AutoModelForCausalLM
 from compute_trainium.execution_isolation import managed_running
 from compute_trainium.training import file_sha256
 from compute_trainium.recoverable_dependencies import pack_dependency
 if managed_running() or os.environ.get('USE_TORCH_XLA')!='0':raise RuntimeError('Closed Native owner and independent CPU required')
 out=root/'dependency-cpu-reference-v1';public=Path('results/recoverable_dependencies_20261028/cpu-reference');out.mkdir(mode=0o700);public.mkdir();start=time.time();torch.set_num_threads(12)
 panel=root/'recoverable-dependency-panel-v2';pr=json.loads((panel/'report.json').read_text());cases=json.loads((panel/'PRIVATE-cases.json').read_text());parent=json.loads((root/'spliced-workspace-discovery-v1/plan.json').read_text())
 if file_sha256(panel/'PRIVATE-cases.json')!=pr['case_file_sha256']:raise AssertionError('Frozen source targets changed')
 indices=[next(i for i,x in enumerate(cases) if x['kind']==kind) for kind in ['code_binding','math_conclusion']]
 plan=dict(identity=out.name,registered_unix=start,seconds_limit=900,reference_panel_indices=indices,panel_sha256=pr['case_file_sha256'],checkpoint_sha256=parent['checkpoint_sha256'],logits_max_abs_limit=.001,real_target_nll_max_abs_limit=.0002,optimizer_updates=0,assigned_chips=0,scope='Independent actual1.485B HF dense2D graph at1024 physical positions.Both full and source-ablated authentic examples,shared original C0,no updates.No panel quality aggregation or selection.')
 for d in [out,public]:(d/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 result=dict(status='STOPPED_DEPENDENCY_CPU_REFERENCE',optimizer_updates=0,assigned_chips=0)
 def guard():
  av=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
  if av<512*1024**3 or shutil.disk_usage(root).free<35*1024**3 or time.time()-start>900 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>64*1024**3:raise RuntimeError('CPU reference reserve/time guard')
 try:
  guard();model=AutoModelForCausalLM.from_pretrained(root/'workspace-olmo2-stage1-base-v1',local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').eval().requires_grad_(False);state=torch.load(root/parent['checkpoint_relative'],weights_only=True,mmap=True)['model'];model.load_state_dict(state);del state;versions=tuple(x._version for x in model.parameters());records=[]
  with torch.no_grad():
   for i in indices:
    for condition in ['full','short']:
     guard();v=pack_dependency(cases[i],condition,model.config.pad_token_id);tokens=v['tokens'];pos=v['positions'];anchors=v['anchors'];n=tokens.shape[1];q=anchors.numel()
     raw=model.model(tokens,position_ids=pos[None],use_cache=False).last_hidden_state;initial=raw[:,anchors];positions=torch.cat([pos,pos[anchors]]);steps=torch.cat([torch.zeros(n),torch.ones(q)])
     allowed=(positions[None,:]<=positions[:,None])&(steps[None,:]<=steps[:,None])&((steps[None,:]==0)|(positions[None,:]==positions[:,None]));mask=torch.zeros(n+q,n+q).masked_fill(~allowed,torch.finfo(torch.float32).min)[None,None]
     refined=model.model(inputs_embeds=torch.cat([model.model.embed_tokens(tokens),initial],1),attention_mask=mask,position_ids=positions[None],use_cache=False).last_hidden_state[:,n:]
     rl=model.lm_head(initial);wl=model.lm_head(.5*initial+.5*refined);target=v['targets'];rp=F.cross_entropy(rl.double().reshape(-1,model.config.vocab_size),target.reshape(-1),reduction='none').reshape_as(target);wp=F.cross_entropy(wl.double().reshape(-1,model.config.vocab_size),target.reshape(-1),reduction='none').reshape_as(target)
     file=out/f'PRIVATE-case{i}-{condition}.pt';torch.save(dict(raw_logits=rl,mixed_logits=wl,raw_nll=rp,workspace_nll=wp,mask=v['mask']),file)
     emb=model.model.embed_tokens(tokens[:,anchors[0]:anchors[0]+1]);records.append(dict(panel_index=i,kind=cases[i]['kind'],condition=condition,relative=str(file.relative_to(root)),sha256=file_sha256(file),token_embedding_rms=float(emb.square().mean().sqrt()),workspace_input_rms=float(initial[:,0].square().mean().sqrt())));(out/'progress.json').write_text(json.dumps(dict(completed=len(records),wall_seconds=time.time()-start)))
  if versions!=tuple(x._version for x in model.parameters()):raise AssertionError('CPU reference changed C0')
  result.update(status='PASS_DEPENDENCY_CPU_REFERENCE',records=records,parameters_unchanged=True)
 except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
 finally:
  result.update(wall_seconds=time.time()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
  for d in [out,public]:(d/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
