"""Prefix-only fixed sparse policy;no thought sampling or target-aware selection."""
import argparse,json,os,time
from pathlib import Path
if 'RESEARCH_VISIBLE_CORES' in os.environ:os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def run(a):
 import torch
 import torch.distributed as dist
 from torch.distributed.device_mesh import init_device_mesh
 from compute_trainium.training import file_sha256
 from compute_trainium.workspace_olmo_candidate import OlmoWorkspaceConfig,rotary_tensors
 from compute_trainium.sparse_workspace_candidate import IndependentSparseWorkspaceCandidate
 from compute_trainium.sparse_workspace_training import pack_view,ANCHORS
 from compute_trainium.workspace_view_batches import WorkspaceViewBatches
 from compute_trainium.scale_model_sharding import shard_model_from_cpu
 from compute_trainium.workspace_native_state import restore_model_shards
 from compute_trainium.stable_target_loss import stable_target_log_probs
 p=json.loads(a.plan.read_text());source=Path(__file__).resolve().parents[1]
 if file_sha256(a.plan)!=a.plan_sha or p['purpose']!='spliced_workspace_three_arm_discovery':raise AssertionError('Frozen discovery evaluation required')
 if any(file_sha256(source/n)!=h for n,h in p['worker_source_sha256'].items()):raise AssertionError('Frozen source changed')
 rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']);started=time.time();torch.set_num_threads(2)
 os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
 import torch_neuronx
 torch.neuron.set_device(0);dist.init_process_group('neuron',rank=rank,world_size=world);mesh=init_device_mesh('neuron',(world,))
 out=a.output;out.mkdir(parents=True,exist_ok=True);cfg=OlmoWorkspaceConfig(**p['model_config'])
 state=torch.load(a.root/p['checkpoint_relative'],weights_only=True,mmap=True)['model']
 with torch.device('meta'):model=IndependentSparseWorkspaceCandidate(cfg).eval()
 model=shard_model_from_cpu(model,mesh,state,'neuron').requires_grad_(False);del state;model.compile_stages('neuron')
 data=WorkspaceViewBatches(a.root/p['data_relative'],'tune');anchors=torch.tensor(ANCHORS,device='neuron');where=torch.arange(127,159,device='neuron');cos,sin=rotary_tensors(cfg,160,'neuron');summary={}
 with torch.no_grad():
  for arm in ['c0','ordinary','latent','constant']:
   if arm!='c0':restore_model_shards(a.root/'runs'/p['arm_runs'][arm]/f'rank{rank}-u{p["updates"][arm]}.zip',model,label='model')
   rows=[]
   for batch in range(p['evaluation_anchors']//world):
    index=batch*world+rank;view=data.at(index);tokens,targets=pack_view(view,cfg.pad_token_id);tokens=tokens.to('neuron');targets=targets.to('neuron')
    if rank==0:(out/'progress.json').write_text(json.dumps(dict(arm=arm,batch=batch,wall_seconds=time.time()-started)))
    result=model.sparse_forward(tokens,cos,sin,anchors,raw_positions=where,mode='constant' if arm=='constant' else 'latent',mix=.5,pause_id=p['pause_token_id'])
    raw=-stable_target_log_probs(result['raw_logits'],targets[:,127:159]).detach().cpu();work=-stable_target_log_probs(result['mixed_logits'],targets.index_select(1,anchors)).detach().cpu();selected=torch.tensor([0,8,16,24]);policy=raw.clone()
    if arm!='ordinary':policy[:,selected]=work
    row=dict(index=index,domain=view['domain'],raw32_nll=float(raw.mean().cpu()),raw_anchor_nll=float(raw[:,selected].mean().cpu()),raw_first_token_nll=float(raw[:,0].mean().cpu()),workspace_anchor_nll=float(work.mean().cpu()),workspace_first_token_nll=float(work[:,0].mean().cpu()),policy32_nll=float(policy.mean().cpu()))
    # Input-content ablation retains the same query count and prefix KV.
    if arm in ['c0','latent']:
     ablated=model.sparse_forward(tokens,cos,sin,anchors,raw_positions=where,mode='constant',mix=.5,pause_id=p['pause_token_id'])
     loss=-stable_target_log_probs(ablated['mixed_logits'],targets.index_select(1,anchors));row['constant_input_anchor_nll']=float(loss.mean().cpu())
    rows.append(row)
   (out/f'rank{rank}-{arm}-losses.json').write_text(json.dumps(rows,indent=2)+'\n');dist.barrier()
   if rank==0:
    all_rows=[v for k in range(world) for v in json.loads((out/f'rank{k}-{arm}-losses.json').read_text())]
    summary[arm]={field:sum(w*sum(v[field] for v in all_rows if v['domain']==d)/sum(v['domain']==d for v in all_rows) for d,w in enumerate([.6,.2,.2])) for field in rows[0] if field.endswith('_nll')}
 fallback=torch_neuronx.get_fallback_ops()
 if fallback:raise AssertionError('Sparse evaluation fallback')
 if rank==0:(out/'report.json').write_text(json.dumps(dict(status='COMPLETE_SPLICED_WORKSPACE_DISCOVERY_EVALUATION',world=world,scores=summary,evaluation_anchors=p['evaluation_anchors'],optimizer_updates=0,fallback_ops=fallback,wall_seconds=time.time()-started,scope='Same448 reused TUNE anchors,one seed. Four fixed sparse opportunities;full32-token deployed policy loss and raw/anchor/first-target metrics all reported.No target-aware selection or allocation. Constant-input ablation can be distribution shifting;trained constant-query arm is the stronger control.No reasoning or from-scratch claim.'),indent=2)+'\n')
 dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus');p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--comm-port',type=int,required=True);run(p.parse_args())
