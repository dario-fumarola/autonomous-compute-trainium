"""All-rank full-model sparse gradient,one-update and exact-state qualification."""
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
 from compute_trainium.scale_model_sharding import shard_model_from_cpu
 from compute_trainium.scale_sharded_candidate import local_parameter
 from compute_trainium.workspace_sharded_optimizer import WorkspaceShardedAdamW,ShardedGradientAccumulator
 from compute_trainium.workspace_native_state import save_state_waves,restore_complete_state,complete_state_tensors,release_compiled_calls
 from compute_trainium.workspace_native_objective import drain_runtime
 from compute_trainium.stable_target_loss import stable_target_log_probs
 p=json.loads(a.plan.read_text());source=Path(__file__).resolve().parents[1]
 if file_sha256(a.plan)!=a.plan_sha or p['purpose']!='sparse_workspace_native_qualification':raise AssertionError('Frozen qualification required')
 if any(file_sha256(source/n)!=h for n,h in p['worker_source_sha256'].items()):raise AssertionError('Source changed')
 rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']);started=time.time();torch.set_num_threads(2)
 os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
 import torch_neuronx
 torch.neuron.set_device(0);dist.init_process_group('neuron',rank=rank,world_size=world);mesh=init_device_mesh('neuron',(world,))
 out=a.output;out.mkdir(parents=True,exist_ok=True)
 def progress(stage):
  if rank==0:(out/'progress.json').write_text(json.dumps(dict(stage=stage,wall_seconds=time.time()-started)))
 progress('loading_sharded_model')
 c0=torch.load(a.root/p['checkpoint_relative'],weights_only=True,mmap=True)['model'];cfg=OlmoWorkspaceConfig(**p['model_config'])
 with torch.device('meta'):model=SparseWorkspaceCandidate(cfg,activation_checkpointing=True).eval()
 model=shard_model_from_cpu(model,mesh,c0,'neuron');del c0
 optimizer=WorkspaceShardedAdamW(model,fused=True);generator=torch.Generator().manual_seed(9499+rank)
 model.compile_stages('neuron');inputs=torch.load(a.root/p['inputs_relative'],weights_only=True)
 tokens,targets,anchors,where=[inputs[k].to('neuron') for k in ['tokens','targets','anchors','raw_positions']]
 cos,sin=rotary_tensors(cfg,160,'neuron');progress('complete_cached_coupled_gradient')
 optimizer.zero_grad();r=model.sparse_forward(tokens,cos,sin,anchors,raw_positions=where)
 sparse_targets=targets.index_select(1,anchors)
 raw=-stable_target_log_probs(r['raw_logits'],targets).mean();work=-stable_target_log_probs(r['mixed_logits'],sparse_targets).mean();loss=.5*raw+.5*work
 scalar_loss=float(loss.detach().cpu());raw_value=float(raw.detach().cpu());work_value=float(work.detach().cpu())
 (loss/world*256).backward();acc=ShardedGradientAccumulator(model);acc.add_and_clear();acc.install();del acc,r,loss,raw,work
 norm=optimizer.finish_gradients(scale=256,clip=1e30);release_compiled_calls(model)
 progress('independent_full_gradient_comparison')
 expected=torch.load(a.root/p['gradient_relative'],weights_only=True,mmap=True);numbers=[]
 def errors(a,b):
  a=a.reshape(-1);b=b.reshape(-1);d=r=0.
  for k in range(0,a.numel(),65536):
   aa=a[k:k+65536].double();bb=b[k:k+65536].double();d+=float((aa-bb).square().sum());r+=float(bb.square().sum())
  return [d,r]
 for name,param in model.named_parameters():numbers.append(errors(local_parameter(param.grad).detach().cpu(),expected[name].chunk(world,0)[rank]))
 v=torch.tensor(numbers,dtype=torch.float32).to('neuron');dist.all_reduce(v);v=v.cpu().double();gradient_errors=(v[:,0]/v[:,1].clamp_min(1e-30)).sqrt();largest_gradient=float(gradient_errors.max());reference_norm=float(v[:,1].sum())
 if largest_gradient>.02:raise AssertionError('Sparse full-gradient family exceeds unchanged2% limit')
 del numbers,v,gradient_errors;progress('actual_qualification_adamw_update')
 optimizer.finish_gradients(scale=1,clip=1.);optimizer.step();optimizer.zero_grad();drain_runtime('neuron')
 if rank==0:(out/'qualification-updates.json').write_text(json.dumps(dict(completed_qualification_updates=1,optimizer_counter=optimizer.update)))
 progress('independent_first_update_and_moments')
 original=torch.load(a.root/p['checkpoint_relative'],weights_only=True,mmap=True)['model']
 clip=min(1.,1/(math.sqrt(reference_norm)+1e-6));numbers=[]
 for index,(name,param) in enumerate(model.named_parameters()):
  before=original[name].chunk(world,0)[rank];cp=torch.nn.Parameter(before.clone());cp.grad=expected[name].chunk(world,0)[rank]*clip
  cpu_optimizer=torch.optim.AdamW([cp],lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=.1,foreach=False,fused=False);cpu_optimizer.step()
  actual=local_parameter(param).detach().cpu();row=errors(actual-before,cp.detach()-before)
  for field in ['exp_avg','exp_avg_sq']:row+=errors(optimizer.adam.state[optimizer.local[index]][field].detach().cpu(),cpu_optimizer.state[cp][field])
  numbers.append(row);del actual,cp,cpu_optimizer;drain_runtime('neuron')
 v=torch.tensor(numbers,dtype=torch.float32).to('neuron');dist.all_reduce(v);v=v.cpu().double();update_errors=(v[:,0]/v[:,1].clamp_min(1e-30)).sqrt();m_errors=(v[:,2]/v[:,3].clamp_min(1e-30)).sqrt();vv_errors=(v[:,4]/v[:,5].clamp_min(1e-30)).sqrt()
 largest_update=float(update_errors.max());largest_moment=max(float(m_errors.max()),float(vv_errors.max()))
 if max(largest_update,largest_moment)>.02:raise AssertionError('Sparse first update or moment family exceeds unchanged2% limit')
 del expected,original,numbers,v,update_errors,m_errors,vv_errors;drain_runtime('neuron')
 progress('post_live_step_complete_state_roundtrip')
 path=out/f'rank{rank}-roundtrip.zip';saved=save_state_waves(path,model,None,optimizer,generator,extra=dict(qualification=True,scientific_optimizer_updates=0),wave_size=2,save_ranks=2)
 if rank<2:
  old_rng=generator.get_state();old_torch_rng=torch.get_rng_state()
  with torch.no_grad():
   for name,value in complete_state_tensors(model,None,optimizer):value.reshape(-1)[0].add_(1.)
  generator.manual_seed(1);torch.manual_seed(2);restored=restore_complete_state(path,model,None,optimizer,generator)
  if restored['update']!=1 or not torch.equal(old_rng,generator.get_state()) or not torch.equal(old_torch_rng,torch.get_rng_state()):raise AssertionError('Complete state or RNG did not restore')
 dist.barrier();fallback=torch_neuronx.get_fallback_ops()
 if fallback:raise AssertionError('Native sparse qualification fallback:'+str(fallback))
 if rank==0:(out/'report.json').write_text(json.dumps(dict(status='PASS_SPARSE_NATIVE_GRADIENT_UPDATE_RECOVERY',world=world,actual_qualification_optimizer_updates=1,scientific_optimizer_updates=0,parameters_checked=len(list(model.parameters())),largest_named_gradient_relative_l2=largest_gradient,largest_named_update_relative_l2=largest_update,largest_named_moment_relative_l2=largest_moment,complete_state_tensors_per_saved_rank=716,roundtrip_ranks=2,exact_rng=True,fallback_ops=fallback,raw159_nll=raw_value,sparse_anchor_nll=work_value,loss=scalar_loss,gradient_norm=norm,wall_seconds=time.time()-started,scope='One authentic TRAIN view repeated over64 ranks for numerical qualification.No scientific learning/quality claim.Two saved shards test all716 actor/AdamW states after a live step;full-global restart and100warm/mature gates remain.')))
 dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus');p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--comm-port',type=int,required=True);run(p.parse_args())
