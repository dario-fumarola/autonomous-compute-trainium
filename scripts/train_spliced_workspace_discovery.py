"""Uniform natural-token NTP;retain exact qualified prefixes without rewinding."""
import argparse,json,os,time,math
from pathlib import Path
if 'RESEARCH_VISIBLE_CORES' in os.environ:os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def run(a):
 import torch
 import torch.distributed as dist
 from torch.distributed.device_mesh import init_device_mesh
 from compute_trainium.training import file_sha256
 from compute_trainium.workspace_olmo_candidate import OlmoWorkspaceConfig,rotary_tensors
 from compute_trainium.sparse_workspace_candidate import IndependentSparseWorkspaceCandidate
 from compute_trainium.sparse_workspace_training import pack_view
 from compute_trainium.spliced_workspace_training import spliced_objective,sample_positions,partition_positions
 from compute_trainium.workspace_view_batches import WorkspaceViewBatches
 from compute_trainium.scale_model_sharding import shard_model_from_cpu
 from compute_trainium.scale_sharded_candidate import local_parameter
 from compute_trainium.workspace_sharded_optimizer import WorkspaceShardedAdamW,ShardedGradientAccumulator
 from compute_trainium.workspace_native_state import save_state_waves,restore_complete_state,release_compiled_calls
 from compute_trainium.workspace_native_objective import drain_runtime
 p=json.loads(a.plan.read_text());source=Path(__file__).resolve().parents[1]
 if file_sha256(a.plan)!=a.plan_sha or p['purpose']!='spliced_workspace_three_arm_discovery' or a.arm not in p['updates']:raise AssertionError('Frozen uniform-NTP arm required')
 if any(file_sha256(source/n)!=h for n,h in p['worker_source_sha256'].items()):raise AssertionError('Frozen source changed')
 rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']);started=time.time();torch.set_num_threads(2)
 os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
 import torch_neuronx
 torch.neuron.set_device(0);dist.init_process_group('neuron',rank=rank,world_size=world);mesh=init_device_mesh('neuron',(world,))
 out=a.output;out.mkdir(parents=True,exist_ok=True)
 def progress(stage,**extra):
  if rank==0:(out/'progress.json').write_text(json.dumps(dict(stage=stage,wall_seconds=time.time()-started,arm=a.arm,**extra)))
 cfg=OlmoWorkspaceConfig(**p['model_config']);c0=torch.load(a.root/p['checkpoint_relative'],weights_only=True,mmap=True)['model']
 with torch.device('meta'):model=IndependentSparseWorkspaceCandidate(cfg,activation_checkpointing=True).eval()
 model=shard_model_from_cpu(model,mesh,c0,'neuron');del c0
 optimizer=WorkspaceShardedAdamW(model,lr=p['lr'],fused=True);generator=torch.Generator().manual_seed(p['sample_seed_base']+rank)
 prefix=p['prefixes'][a.arm];start=prefix['updates']
 if start:
  progress('restore_complete_retained_prefix',updates=start)
  path=a.root/prefix['run_relative']/f'rank{rank}-u{start}.zip'
  if file_sha256(path)!=prefix['checkpoint_sha256'][rank]:raise AssertionError('Frozen retained prefix bytes changed')
  restored=restore_complete_state(path,model,None,optimizer,generator)
  if restored['update']!=start or restored['data_cursor']!=start*world:raise AssertionError('Exact source cursor and update counter required')
  for group in optimizer.adam.param_groups:
   if group['lr']!=p['lr'] or tuple(group['betas'])!=(.9,.95) or group['eps']!=1e-8 or group['weight_decay']!=.1:raise AssertionError('No optimizer recipe change on continuation')
 data=WorkspaceViewBatches(a.root/p['data_relative'],'train');cos,sin=rotary_tensors(cfg,160,'neuron');model.compile_stages('neuron');records=[]
 def backward(index,*,numerical=False):
  tokens,targets=pack_view(data.at(index),cfg.pad_token_id);tokens=tokens.to('neuron');targets=targets.to('neuron');optimizer.zero_grad()
  selected,remaining=partition_positions(torch.tensor(p['mature_anchors'])) if numerical else sample_positions(generator,count=32)
  loss,raw,anchor=spliced_objective(model,tokens,targets,cos,sin,selected.to('neuron'),remaining.to('neuron'),mode=a.arm,pause_id=p['pause_token_id']);values=[float(v.detach().cpu()) for v in [loss,raw,anchor]]
  (loss/world*256).backward();acc=ShardedGradientAccumulator(model);acc.add_and_clear();acc.install();del acc,loss,raw,anchor
  return values
 if a.arm=='latent':
  progress('exact_retained_prefix_gradient_replay');backward(prefix['mature_audit_index'],numerical=True);optimizer.finish_gradients(scale=256,clip=1e30)
  expected=torch.load(a.root/prefix['run_relative']/f'rank{rank}-mature-gradients.pt',weights_only=True,mmap=True);maximum=max(float((local_parameter(v.grad).detach().cpu()-expected[n]).abs().max()) for n,v in model.named_parameters());del expected
  summary=torch.tensor([maximum],device='neuron');dist.all_reduce(summary,op=dist.ReduceOp.MAX);maximum=float(summary.cpu()[0])
  if maximum!=0:raise AssertionError('Retained latent prefix replay must remain exact before update102')
  if rank==0:(out/'prefix-replay.json').write_text(json.dumps(dict(status='PASS_EXACT_PREFIX_REPLAY',max_abs_error=maximum,optimizer_updates=0,counter=start)))
  optimizer.zero_grad()
 elif a.arm=='constant':
  progress('cold_full179_constant_gradient_gate');backward(0,numerical=True);optimizer.finish_gradients(scale=256,clip=1e30)
  reference=p['constant_reference'];path=a.root/reference['gradient_relative']
  if file_sha256(path)!=reference['gradient_sha256']:raise AssertionError('Constant CPU oracle changed')
  expected=torch.load(path,weights_only=True,mmap=True);numbers=[]
  for n,v in model.named_parameters():
   have=local_parameter(v.grad).detach().cpu().double();wanted=expected[n].chunk(world,0)[rank].double();numbers.append([float((have-wanted).square().sum()),float(wanted.square().sum())])
  comparison=torch.tensor(numbers,dtype=torch.float32,device='neuron');dist.all_reduce(comparison);comparison=comparison.cpu().double();maximum=float((comparison[:,0]/comparison[:,1].clamp_min(1e-30)).sqrt().max())
  if maximum>.02:raise AssertionError('Constant cold179-gradient gate exceeds unchanged2% limit')
  if rank==0:(out/'cold-gradient-gate.json').write_text(json.dumps(dict(status='PASS_CONSTANT_FULL179_GRADIENT_GATE',largest_named_relative_l2=maximum,optimizer_updates=0)))
  optimizer.zero_grad();del expected,numbers,comparison
 for update in range(start+1,p['updates'][a.arm]+1):
  progress('learning',update=update,total_updates=p['updates'][a.arm]);tick=time.perf_counter();values=backward((update-1)*world+rank)
  norm=optimizer.finish_gradients(scale=256);optimizer.step();optimizer.zero_grad();drain_runtime('neuron')
  if optimizer.update!=update:raise AssertionError('No reset or hidden updates')
  global_values=torch.tensor(values,dtype=torch.float32,device='neuron');dist.all_reduce(global_values);values=(global_values/world).cpu().tolist()
  row=dict(update=update,optimizer_counter=optimizer.update,lr=p['lr'],seconds=time.perf_counter()-tick,loss=values[0],raw_subset_nll=values[1],anchor_nll=values[2],gradient_norm=norm);records.append(row)
  if rank==0:(out/'updates.json').write_text(json.dumps(records,indent=2)+'\n')
  if not all(math.isfinite(v) for v in [*values,norm]):raise FloatingPointError('Nonfinite uniform-NTP update')
  if len(records)==6 and sum(v['seconds'] for v in records[1:])/5>8:raise RuntimeError('Registered8-second warm stability screen failed')
 progress('mature_zero_update_audit_gradient');values=backward(p['mature_audit_index'],numerical=True);norm=optimizer.finish_gradients(scale=256,clip=1e30);release_compiled_calls(model)
 torch.save({n:local_parameter(v.grad).detach().cpu() for n,v in model.named_parameters()},out/f'rank{rank}-mature-gradients.pt')
 if rank==0:torch.save(data.at(p['mature_audit_index']),out/'PRIVATE-mature-view.pt')
 optimizer.zero_grad();drain_runtime('neuron');progress('complete_all64_endpoint_checkpoint');total=p['updates'][a.arm]
 saved=save_state_waves(out/f'rank{rank}-u{total}.zip',model,None,optimizer,generator,extra=dict(plan_sha256=a.plan_sha,data_cursor=total*world,arm=a.arm,scientific_discovery=True,uniform_ntp=True,retained_prefix_updates=start,source_report_sha256=p['data_report_sha256']),wave_size=2)
 dist.barrier();fallback=torch_neuronx.get_fallback_ops()
 if fallback:raise AssertionError('Uniform-NTP fallback:'+str(fallback))
 if rank==0:
  result=dict(status='COMPLETE_SPARSE_DISCOVERY_ARM_PENDING_MATURE_HF_AUDIT',world=world,arm=a.arm,actual_optimizer_updates=total,retained_prefix_updates=start,new_optimizer_updates=total-start,source_views=total*world,natural_label_exposures=total*world*159,nominal_matrix_operations=total*world*p['nominal_operations_per_view'][a.arm],first_new_update_seconds=records[0]['seconds'],warm_mean_seconds=sum(v['seconds'] for v in records[1:])/max(1,len(records)-1),wall_seconds=time.time()-started,checkpoint_bytes_per_rank=saved['bytes'],mature_loss=values[0],mature_gradient_norm=norm,fallback_ops=fallback,heldout_read=False,scope='All authentic labels uniformly weighted once.No auxiliary loss,PG,teacher or new parameters.Retained prefix counts included in total;costs reported separately.Short constant-rate single-seed discovery,not tuned convergence,learned allocation,reasoning or random initialization.')
  (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
 dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus');p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--comm-port',type=int,required=True);p.add_argument('--arm',choices=['ordinary','latent','constant'],required=True);run(p.parse_args())
