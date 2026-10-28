"""Fresh-C0 bounded differentiable-workspace discovery arm."""
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
 from compute_trainium.workspace_native_state import save_state_waves,release_compiled_calls
 from compute_trainium.workspace_native_objective import drain_runtime
 p=json.loads(a.plan.read_text());source=Path(__file__).resolve().parents[1]
 if file_sha256(a.plan)!=a.plan_sha or p['purpose']!='sparse_workspace_three_arm_discovery' or a.arm not in p['updates']:raise AssertionError('Frozen discovery arm required')
 if any(file_sha256(source/n)!=h for n,h in p['worker_source_sha256'].items()):raise AssertionError('Frozen source changed')
 rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']);started=time.time();torch.set_num_threads(2)
 os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
 import torch_neuronx
 torch.neuron.set_device(0);dist.init_process_group('neuron',rank=rank,world_size=world);mesh=init_device_mesh('neuron',(world,))
 out=a.output;out.mkdir(parents=True,exist_ok=True)
 def progress(stage,**extra):
  if rank==0:(out/'progress.json').write_text(json.dumps(dict(stage=stage,wall_seconds=time.time()-started,arm=a.arm,**extra)))
 cfg=OlmoWorkspaceConfig(**p['model_config']);c0=torch.load(a.root/p['checkpoint_relative'],weights_only=True,mmap=True)['model']
 with torch.device('meta'):model=SparseWorkspaceCandidate(cfg,activation_checkpointing=True).eval()
 model=shard_model_from_cpu(model,mesh,c0,'neuron');del c0
 optimizer=WorkspaceShardedAdamW(model,lr=p['lr'],fused=True);generator=torch.Generator().manual_seed(p['seed']+rank)
 data=WorkspaceViewBatches(a.root/p['data_relative'],'train');anchors=torch.tensor(ANCHORS,device='neuron');cos,sin=rotary_tensors(cfg,160,'neuron');model.compile_stages('neuron');records=[]
 def backward(index):
  tokens,targets=pack_view(data.at(index),cfg.pad_token_id);tokens=tokens.to('neuron');targets=targets.to('neuron');optimizer.zero_grad()
  loss,raw,anchor=sparse_objective(model,tokens,targets,cos,sin,anchors,mode=a.arm,pause_id=p['pause_token_id']);values=[float(v.detach().cpu()) for v in [loss,raw,anchor]]
  (loss/world*256).backward();acc=ShardedGradientAccumulator(model);acc.add_and_clear();acc.install();del acc,loss,raw,anchor
  return values
 for update in range(1,p['updates'][a.arm]+1):
  progress('learning',update=update,total_updates=p['updates'][a.arm]);tick=time.perf_counter();values=backward((update-1)*world+rank)
  norm=optimizer.finish_gradients(scale=256);optimizer.step();optimizer.zero_grad();drain_runtime('neuron')
  global_values=torch.tensor(values,dtype=torch.float32,device='neuron');dist.all_reduce(global_values);values=(global_values/world).cpu().tolist()
  row=dict(update=update,optimizer_counter=optimizer.update,lr=p['lr'],seconds=time.perf_counter()-tick,loss=values[0],raw159_nll=values[1],anchor_nll=values[2],gradient_norm=norm);records.append(row)
  if rank==0:(out/'updates.json').write_text(json.dumps(records,indent=2)+'\n')
  if not all(math.isfinite(v) for v in [*values,norm]):raise FloatingPointError('Nonfinite sparse discovery')
 progress('mature_zero_update_audit_gradient');values=backward(p['mature_audit_index']);norm=optimizer.finish_gradients(scale=256,clip=1e30);release_compiled_calls(model)
 torch.save({n:local_parameter(v.grad).detach().cpu() for n,v in model.named_parameters()},out/f'rank{rank}-mature-gradients.pt')
 if rank==0:torch.save(data.at(p['mature_audit_index']),out/'PRIVATE-mature-view.pt')
 optimizer.zero_grad();drain_runtime('neuron');progress('complete_all64_endpoint_checkpoint');total=p['updates'][a.arm]
 saved=save_state_waves(out/f'rank{rank}-u{total}.zip',model,None,optimizer,generator,extra=dict(plan_sha256=a.plan_sha,data_cursor=total*world,arm=a.arm,scientific_discovery=True,source_report_sha256=p['data_report_sha256']),wave_size=p['checkpoint_writer_wave_size'])
 dist.barrier();fallback=torch_neuronx.get_fallback_ops()
 if fallback:raise AssertionError('Sparse discovery fallback:'+str(fallback))
 if rank==0:
  result=dict(status='COMPLETE_SPARSE_DISCOVERY_ARM_PENDING_MATURE_HF_AUDIT',world=world,arm=a.arm,actual_optimizer_updates=total,source_views=total*world,raw_label_exposures=total*world*159,anchor_label_exposures=total*world*4,nominal_matrix_operations=total*world*sparse_matrix_work(cfg,a.arm),first_update_seconds=records[0]['seconds'],warm_mean_seconds=sum(v['seconds'] for v in records[1:])/len(records[1:]),wall_seconds=time.time()-started,checkpoint_bytes_per_rank=saved['bytes'],mature_loss=values[0],mature_gradient_norm=norm,fallback_ops=fallback,heldout_read=False,scope='Single-seed constant-rate discovery from commonC0 with freshAdamW.Same natural targets,source order and .5 raw159+.5 anchor4 emphasis in every arm.Nominal matrix work matched,not hardware time or converged tuned efficiency.No allocation,reasoning or from-scratch claim.')
  (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
 dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus');p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--comm-port',type=int,required=True);p.add_argument('--arm',choices=['ordinary','latent','constant'],required=True);run(p.parse_args())
