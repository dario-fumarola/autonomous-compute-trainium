"""Frozen C0 final states;the existing160-position Native forward only."""
import os,argparse,json,time
from pathlib import Path
if 'RESEARCH_VISIBLE_CORES' in os.environ:os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def run(a):
 import torch
 import torch.distributed as dist
 from torch.distributed.device_mesh import init_device_mesh
 from compute_trainium.workspace_olmo_candidate import WorkspaceOlmoCandidate,OlmoWorkspaceConfig,rotary_tensors
 from compute_trainium.scale_model_sharding import shard_model_from_cpu
 from compute_trainium.workspace_view_batches import WorkspaceViewBatches
 from compute_trainium.sparse_workspace_training import pack_view
 from compute_trainium.training import file_sha256
 p=json.loads(a.plan.read_text());source=Path(__file__).resolve().parents[1]
 if file_sha256(a.plan)!=a.plan_sha or any(file_sha256(source/n)!=h for n,h in p['worker_source_sha256'].items()):raise AssertionError('Frozen feature extraction required')
 rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']);start=time.time();torch.set_num_threads(2);os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
 import torch_neuronx
 torch.neuron.set_device(0);dist.init_process_group('neuron',rank=rank,world_size=world);mesh=init_device_mesh('neuron',(world,));out=a.output;out.mkdir(parents=True,exist_ok=True);cfg=OlmoWorkspaceConfig(**p['model_config']);state=torch.load(a.root/p['checkpoint_relative'],weights_only=True,mmap=True)['model']
 with torch.device('meta'):model=WorkspaceOlmoCandidate(cfg).eval()
 model=shard_model_from_cpu(model,mesh,state,'neuron').requires_grad_(False);del state;model.compile_stages('neuron');cos,sin=rotary_tensors(cfg,160,'neuron');versions=tuple(v._version for v in model.parameters())
 def features(tokens):
  x=model.model.embed_tokens(tokens)
  for layer in model.model.layers:x=layer(x,cos,sin)
  return model.model.norm(x)
 with torch.no_grad():
  errors=[];data=WorkspaceViewBatches(a.root/p['data_relative'],'train')
  for ref in p['cpu_references']:
   if file_sha256(a.root/ref['relative'])!=ref['sha256']:raise AssertionError('Independent HF reference changed')
   expected=torch.load(a.root/ref['relative'],weights_only=True);tokens,targets=pack_view(data.at(ref['index']),cfg.pad_token_id);actual=features(tokens.to('neuron'));logits=model.lm_head(actual[:,127:159]);e=torch.tensor([float((actual.cpu()-expected['features']).abs().max()),float((logits.cpu()-expected['logits']).abs().max())]).to('neuron');dist.all_reduce(e,op=dist.ReduceOp.MAX);values=e.cpu().tolist();errors.append(values)
   if values[0]>.001 or values[1]>.001:raise AssertionError('Frozen feature CPU/Native gate failed')
  for split,count in [('train',p['train_views']),('tune',p['evaluation_anchors'])]:
   data=WorkspaceViewBatches(a.root/p['data_relative'],split);hs=[];ys=[];domains=[];indices=[]
   for batch in range(count//world):
    index=batch*world+rank;view=data.at(index);tokens,targets=pack_view(view,cfg.pad_token_id);hs.append(features(tokens.to('neuron'))[:,:159].cpu());ys.append(targets);domains.append(view['domain']);indices.append(index)
    if rank==0:(out/'progress.json').write_text(json.dumps(dict(split=split,batch=batch,wall_seconds=time.time()-start)))
   torch.save(dict(features=torch.cat(hs),targets=torch.cat(ys),domains=torch.tensor(domains),indices=torch.tensor(indices)),out/f'PRIVATE-rank{rank}-{split}.pt');dist.barrier()
  if versions!=tuple(v._version for v in model.parameters()):raise AssertionError('Feature extraction changed C0')
 fallback=torch_neuronx.get_fallback_ops()
 if fallback:raise AssertionError('Feature extraction fallback:'+str(fallback))
 if rank==0:(out/'report.json').write_text(json.dumps(dict(status='COMPLETE_FROZEN_RESIDUAL_FEATURES',world=world,optimizer_updates=0,feature_dtype='float32',train_views=p['train_views'],evaluation_views=p['evaluation_anchors'],cpu_errors=errors,parameters_unchanged=True,fallback_ops=fallback,wall_seconds=time.time()-start),indent=2)+'\n')
 dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus');p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--comm-port',type=int,required=True);run(p.parse_args())
