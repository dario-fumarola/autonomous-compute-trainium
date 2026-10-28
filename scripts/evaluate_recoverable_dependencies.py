"""Source-frozen dependency assay;zero updates and no answer-aware routing."""
import argparse,json,os,time,math
from pathlib import Path
if 'RESEARCH_VISIBLE_CORES' in os.environ:os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def run(a):
 import torch
 import torch.distributed as dist
 from torch.distributed.device_mesh import init_device_mesh
 from compute_trainium.training import file_sha256
 from compute_trainium.workspace_olmo_candidate import OlmoWorkspaceConfig,rotary_tensors
 from compute_trainium.dependency_candidate import DependencyCandidate
 from compute_trainium.recoverable_dependencies import pack_dependency,dependency_metrics
 from compute_trainium.scale_model_sharding import shard_model_from_cpu
 from compute_trainium.workspace_native_state import restore_model_shards
 from compute_trainium.stable_target_loss import stable_target_log_probs
 p=json.loads(a.plan.read_text());source=Path(__file__).resolve().parents[1]
 if file_sha256(a.plan)!=a.plan_sha or p['purpose']!='recoverable_dependency_inference':raise AssertionError('Frozen dependency assay required')
 if any(file_sha256(source/n)!=h for n,h in p['worker_source_sha256'].items()):raise AssertionError('Frozen source changed')
 if bool(a.diagnose_only)!=bool(p.get('diagnostic_only',False)):raise AssertionError('Diagnosis mode must be separately registered')
 rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']);start=time.time();torch.set_num_threads(2);os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
 import torch_neuronx
 torch.neuron.set_device(0);dist.init_process_group('neuron',rank=rank,world_size=world);mesh=init_device_mesh('neuron',(world,));out=a.output;out.mkdir(parents=True,exist_ok=True)
 cfg=OlmoWorkspaceConfig(**p['model_config']);state=torch.load(a.root/p['checkpoint_relative'],weights_only=True,mmap=True)['model']
 with torch.device('meta'):model=DependencyCandidate(cfg).eval()
 model=shard_model_from_cpu(model,mesh,state,'neuron').requires_grad_(False);del state;model.compile_stages('neuron');cos_table,sin_table=rotary_tensors(cfg,2048,'neuron')
 if file_sha256(a.root/p['panel_relative'])!=p['panel_sha256']:raise AssertionError('Frozen natural source panel changed')
 cases=json.loads((a.root/p['panel_relative']).read_text());references=[]
 def forward(index,condition,mode='latent',variant='original'):
  case=cases[index];v=pack_dependency(case,condition,cfg.pad_token_id);positions=v['positions'].to('neuron');cos=cos_table.index_select(-2,positions);sin=sin_table.index_select(-2,positions);anchors=v['anchors'].to('neuron')
  r=model.dependency_forward(v['tokens'].to('neuron'),cos,sin,anchors,raw_positions=anchors,mode=mode,pause_id=p['pause_token_id'],variant=variant)
  return v,r
 with torch.no_grad():
  # Every rank checks the same four authentic examples before any panel score.
  for reference in p['cpu_references']+p['identity_references']:
   if rank==0:(out/'progress.json').write_text(json.dumps(dict(stage='independent_full_size_cpu_gate',reference=len(references),wall_seconds=time.time()-start)))
   v,r=forward(reference['panel_index'],reference['condition'],variant=reference.get('variant','original'));expected=torch.load(a.root/reference['relative'],weights_only=True,mmap=True)
   if file_sha256(a.root/reference['relative'])!=reference['sha256']:raise AssertionError('Independent CPU oracle changed')
   if reference.get('variant')=='embedding_replay':expected=dict(expected,mixed_logits=expected['raw_logits'],workspace_nll=expected['raw_nll'])
   raw=r['raw_logits'].detach().cpu();worked=r['mixed_logits'].detach().cpu();rn=-stable_target_log_probs(raw,v['targets']);wn=-stable_target_log_probs(worked,v['targets']);mask=v['mask']
   errors=[float((raw-expected['raw_logits']).abs().max()),float((worked-expected['mixed_logits']).abs().max()),float((rn.double()[mask]-expected['raw_nll'][mask]).abs().max()),float((wn.double()[mask]-expected['workspace_nll'][mask]).abs().max())]
   collective=torch.tensor(errors,dtype=torch.float32).to('neuron');dist.all_reduce(collective,op=dist.ReduceOp.MAX);errors=collective.cpu().tolist()
   real=v['mask'][0];real_errors=[float((raw-expected['raw_logits']).abs().amax(-1)[0,real].max()),float((worked-expected['mixed_logits']).abs().amax(-1)[0,real].max())]
   record=dict(panel_index=reference['panel_index'],condition=reference['condition'],variant=reference.get('variant','original'),errors=errors,real_query_logits_max_abs=real_errors,per_query_raw_logits_max_abs=(raw-expected['raw_logits']).abs().amax(-1).tolist(),per_query_workspace_logits_max_abs=(worked-expected['mixed_logits']).abs().amax(-1).tolist(),real_query_mask=v['mask'].tolist())
   if rank==0:
    (out/f'forward-comparison-{len(references)}.json').write_text(json.dumps(record,indent=2)+'\n')
    if a.diagnose_only:torch.save(dict(raw_logits=raw,mixed_logits=worked),out/f'PRIVATE-native-reference-{len(references)}.pt')
   if not a.diagnose_only and (max(errors[:2])>.001 or max(errors[2:])>.0002):raise AssertionError('New1024-position Native forward fails independent CPU tolerance')
   references.append(dict(panel_index=reference['panel_index'],condition=reference['condition'],variant=reference.get('variant','original'),raw_logits_max_abs=errors[0],workspace_logits_max_abs=errors[1],raw_target_nll_max_abs=errors[2],workspace_target_nll_max_abs=errors[3]))
  if a.diagnose_only:
   dist.barrier();fallback=torch_neuronx.get_fallback_ops()
   if rank==0:(out/'report.json').write_text(json.dumps(dict(status='COMPLETE_DEPENDENCY_FORWARD_DIAGNOSIS',world=world,optimizer_updates=0,records=references,fallback_ops=fallback,wall_seconds=time.time()-start,scope='Numerical failure diagnosis only.No panel scoring,no gate waiver and no scientific quality claim.'),indent=2)+'\n')
   dist.barrier();dist.destroy_process_group();return
  if rank==0:(out/'cpu-forward-gate.json').write_text(json.dumps(dict(status='PASS_FULL_SIZE_DEPENDENCY_FORWARD_GATE',optimizer_updates=0,records=references),indent=2)+'\n')
  summary={};calls=0
  for arm in ['c0','ordinary','latent','constant']:
   if arm!='c0':restore_model_shards(a.root/'runs'/p['arm_runs'][arm]/f'rank{rank}-u{p["updates"][arm]}.zip',model,label='model')
   versions=tuple(v._version for v in model.parameters());rows=[]
   for variant,condition in [('original','full'),('original','short'),('strict','full'),('strict_rms','full')]:
    for batch in range(math.ceil(len(cases)/world)):
     index=batch*world+rank;safe=index if index<len(cases) else 0
     if rank==0:(out/'progress.json').write_text(json.dumps(dict(stage='frozen_panel_scoring',arm=arm,variant=variant,condition=condition,batch=batch,wall_seconds=time.time()-start)))
     v,r=forward(safe,condition,'constant' if arm=='constant' else 'latent',variant=variant);calls+=1
     metrics=dependency_metrics(r['raw_logits'],r['mixed_logits'],v['targets'].to('neuron'),v['mask'].to('neuron'),r['initial_workspace'],r['workspace_state']);values={k:float(x.detach().cpu()) for k,x in metrics.items()}
     if not all(math.isfinite(x) for x in values.values()):raise FloatingPointError('Nonfinite dependency metric')
     if index<len(cases):rows.append(dict(index=index,id=cases[index]['id'],kind=cases[index]['kind'],cohort=cases[index]['cohort'],group=cases[index]['group'],condition=condition,variant=variant,**values))
   if versions!=tuple(v._version for v in model.parameters()):raise AssertionError('Zero-update assay changed model parameters')
   (out/f'rank{rank}-{arm}-metrics.json').write_text(json.dumps(rows,indent=2)+'\n');dist.barrier()
   if rank==0:
    combined=[x for k in range(world) for x in json.loads((out/f'rank{k}-{arm}-metrics.json').read_text())];summary[arm]=dict(real_case_condition_rows=len(combined),parameters_unchanged=True)
 fallback=torch_neuronx.get_fallback_ops()
 if fallback:raise AssertionError('Dependency assay fallback:'+str(fallback))
 if rank==0:(out/'report.json').write_text(json.dumps(dict(status='COMPLETE_RECOVERABLE_DEPENDENCY_INFERENCE',world=world,optimizer_updates=0,case_count=len(cases),physical_case_condition_forwards=calls*world,numerical_qualification_forwards=len(references)*world,models=summary,cpu_forward_gate=references,parameters_unchanged=True,fallback_ops=fallback,wall_seconds=time.time()-start,scope='Original targets only.No fitting,prompt rewrite,mixer sweep,extra latent depth or target-aware selection.Long code context extrapolates beyond continuation length;short views are content ablations,recomputing all states/KV.Development mechanism diagnostic,not fresh confirmation or training-efficiency claim.'),indent=2)+'\n')
 dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus');p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--comm-port',type=int,required=True);p.add_argument('--diagnose-only',action='store_true');run(p.parse_args())
