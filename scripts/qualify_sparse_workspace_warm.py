"""One cold+100warm actual sparse updates and a fresh-process state replay."""
import argparse,json,os,time,math
from pathlib import Path
if 'RESEARCH_VISIBLE_CORES' in os.environ:os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def run(a):
 import torch
 import torch.distributed as dist
 from torch.distributed.device_mesh import init_device_mesh
 from compute_trainium.training import file_sha256
 from compute_trainium.workspace_olmo_candidate import OlmoWorkspaceConfig,rotary_tensors
 from compute_trainium.sparse_workspace_candidate import SparseWorkspaceCandidate
 from compute_trainium.sparse_workspace_training import sparse_objective,pack_view,sparse_matrix_work,ANCHORS
 from compute_trainium.workspace_view_batches import WorkspaceViewBatches
 from compute_trainium.scale_model_sharding import shard_model_from_cpu
 from compute_trainium.scale_sharded_candidate import local_parameter
 from compute_trainium.workspace_sharded_optimizer import WorkspaceShardedAdamW,ShardedGradientAccumulator
 from compute_trainium.workspace_native_state import save_state_waves,restore_complete_state,release_compiled_calls
 from compute_trainium.workspace_native_objective import drain_runtime
 p=json.loads(a.plan.read_text());source=Path(__file__).resolve().parents[1]
 if file_sha256(a.plan)!=a.plan_sha or p['purpose']!='sparse_workspace_warm_and_replay_qualification':raise AssertionError('Frozen sparse qualification required')
 if any(file_sha256(source/n)!=h for n,h in p['worker_source_sha256'].items()):raise AssertionError('Frozen source changed')
 rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']);started=time.time();torch.set_num_threads(2)
 os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
 import torch_neuronx
 torch.neuron.set_device(0);dist.init_process_group('neuron',rank=rank,world_size=world);mesh=init_device_mesh('neuron',(world,))
 out=a.output;out.mkdir(parents=True,exist_ok=True)
 def progress(stage,**extra):
  if rank==0:(out/'progress.json').write_text(json.dumps(dict(stage=stage,wall_seconds=time.time()-started,**extra)))
 cfg=OlmoWorkspaceConfig(**p['model_config']);c0=torch.load(a.root/p['checkpoint_relative'],weights_only=True,mmap=True)['model']
 with torch.device('meta'):model=SparseWorkspaceCandidate(cfg,activation_checkpointing=True).eval()
 model=shard_model_from_cpu(model,mesh,c0,'neuron');del c0
 optimizer=WorkspaceShardedAdamW(model,lr=1e-5,fused=True);generator=torch.Generator().manual_seed(9821+rank)
 data=WorkspaceViewBatches(a.root/p['data_relative'],'train');anchors=torch.tensor(ANCHORS,device='neuron');cos,sin=rotary_tensors(cfg,160,'neuron')
 if a.mode=='replay':
  progress('fresh_process_complete_global_restore')
  restored=restore_complete_state(a.root/'runs'/p['warm_phase']/f'rank{rank}-u101.zip',model,None,optimizer,generator)
  if restored['update']!=101 or restored['data_cursor']!=101*world:raise AssertionError('Exact optimizer and source cursor required')
 model.compile_stages('neuron')
 def backward(index):
  view=data.at(index);tokens,targets=pack_view(view,cfg.pad_token_id);tokens=tokens.to('neuron');targets=targets.to('neuron')
  optimizer.zero_grad();loss,raw,anchor=sparse_objective(model,tokens,targets,cos,sin,anchors)
  values=[float(v.detach().cpu()) for v in [loss,raw,anchor]]
  (loss/world*256).backward();acc=ShardedGradientAccumulator(model);acc.add_and_clear();acc.install();del acc,loss,raw,anchor
  return values
 if a.mode=='warm':
  records=[]
  for update in range(1,102):
   progress('qualifying_updates',update=update);tick=time.perf_counter();values=backward((update-1)*world+rank)
   norm=optimizer.finish_gradients(scale=256);optimizer.step();optimizer.zero_grad();drain_runtime('neuron')
   scalar=torch.tensor(values,dtype=torch.float32,device='neuron');dist.all_reduce(scalar);values=(scalar/world).cpu().tolist()
   row=dict(update=update,optimizer_counter=optimizer.update,seconds=time.perf_counter()-tick,loss=values[0],raw159_nll=values[1],anchor_nll=values[2],gradient_norm=norm);records.append(row)
   if rank==0:(out/'updates.json').write_text(json.dumps(records,indent=2)+'\n')
   if not all(math.isfinite(v) for v in [*values,norm]):raise FloatingPointError('Nonfinite sparse update')
  progress('mature_zero_update_gradient')
  values=backward(p['mature_audit_index']);norm=optimizer.finish_gradients(scale=256,clip=1e30);release_compiled_calls(model)
  torch.save({n:local_parameter(v.grad).detach().cpu() for n,v in model.named_parameters()},out/f'rank{rank}-mature-gradients.pt')
  if rank==0:torch.save(data.at(p['mature_audit_index']),out/'PRIVATE-mature-view.pt')
  optimizer.zero_grad();drain_runtime('neuron');progress('all64_complete_state_save')
  saved=save_state_waves(out/f'rank{rank}-u101.zip',model,None,optimizer,generator,extra=dict(plan_sha256=a.plan_sha,data_cursor=101*world,qualification=True),wave_size=1)
  result=dict(status='COMPLETE_SPARSE_WARM_PENDING_MATURE_AUDIT',world=world,actual_qualification_optimizer_updates=101,scientific_optimizer_updates=0,warm_updates=100,first_update_seconds=records[0]['seconds'],warm_mean_seconds=sum(v['seconds'] for v in records[1:])/100,nominal_matrix_operations=101*world*sparse_matrix_work(cfg),source_views=101*world,raw_label_exposures=101*world*159,anchor_label_exposures=101*world*4,checkpoint_bytes_per_rank=saved['bytes'],mature_loss=values[0],mature_gradient_norm=norm)
 else:
  progress('zero_update_recomputed_gradient_replay');values=backward(p['mature_audit_index']);norm=optimizer.finish_gradients(scale=256,clip=1e30)
  expected=torch.load(a.root/'runs'/p['warm_phase']/f'rank{rank}-mature-gradients.pt',weights_only=True,mmap=True);largest=0.
  for name,v in model.named_parameters():largest=max(largest,float((local_parameter(v.grad).detach().cpu()-expected[name]).abs().max()))
  summary=torch.tensor([largest],device='neuron');dist.all_reduce(summary,op=dist.ReduceOp.MAX);largest=float(summary.cpu()[0])
  if largest!=0:raise AssertionError('Fresh-process sparse gradient replay is not exact')
  result=dict(status='PASS_SPARSE_FULL_GLOBAL_RESTORE_AND_GRADIENT_REPLAY',world=world,actual_qualification_optimizer_updates=0,scientific_optimizer_updates=0,restored_optimizer_counter=optimizer.update,all_rank_complete_state_hashes_verified=True,all_rank_rng_exact=True,replayed_gradient_max_abs_error=largest,recomputed_loss=values[0])
 dist.barrier();fallback=torch_neuronx.get_fallback_ops()
 if fallback:raise AssertionError('Sparse qualification used fallback:'+str(fallback))
 if rank==0:result.update(wall_seconds=time.time()-started,fallback_ops=fallback);(out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
 dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus');p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--comm-port',type=int,required=True);p.add_argument('--mode',choices=['warm','replay'],required=True);run(p.parse_args())
