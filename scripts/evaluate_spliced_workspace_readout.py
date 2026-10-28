"""Frozen models,global TRAIN calibration,then unchanged development readout."""
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
 from compute_trainium.sparse_workspace_training import pack_view
 from compute_trainium.workspace_view_batches import WorkspaceViewBatches
 from compute_trainium.scale_model_sharding import shard_model_from_cpu
 from compute_trainium.workspace_native_state import restore_model_shards
 from compute_trainium.spliced_workspace_readout import readout_grid,choose_readout
 p=json.loads(a.plan.read_text());source=Path(__file__).resolve().parents[1]
 if file_sha256(a.plan)!=a.plan_sha or p['purpose']!='spliced_workspace_global_train_readout':raise AssertionError('Frozen readout registration required')
 if any(file_sha256(source/n)!=h for n,h in p['worker_source_sha256'].items()):raise AssertionError('Frozen source changed')
 rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']);started=time.time();torch.set_num_threads(2)
 os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
 import torch_neuronx
 torch.neuron.set_device(0);dist.init_process_group('neuron',rank=rank,world_size=world);mesh=init_device_mesh('neuron',(world,))
 out=a.output;out.mkdir(parents=True,exist_ok=True);cfg=OlmoWorkspaceConfig(**p['model_config']);state=torch.load(a.root/p['checkpoint_relative'],weights_only=True,mmap=True)['model']
 with torch.device('meta'):model=IndependentSparseWorkspaceCandidate(cfg).eval()
 model=shard_model_from_cpu(model,mesh,state,'neuron').requires_grad_(False);del state;model.compile_stages('neuron')
 anchors=torch.tensor([127,135,143,151],device='neuron');selected=torch.tensor([0,8,16,24],device='neuron');where=torch.arange(127,159,device='neuron');cos,sin=rotary_tensors(cfg,160,'neuron');summary={};choices={};raw_parity={}
 with torch.no_grad():
  for arm in ['c0','ordinary','latent','constant']:
   if arm!='c0':restore_model_shards(a.root/'runs'/p['arm_runs'][arm]/f'rank{rank}-u{p["updates"][arm]}.zip',model,label='model')
   versions=tuple(v._version for v in model.parameters());choice=None
   for split,count in [('train',p['calibration_views']),('tune',p['evaluation_anchors'])]:
    data=WorkspaceViewBatches(a.root/p['data_relative'],split);rows=[]
    for batch in range(count//world):
     index=batch*world+rank;view=data.at(index);tokens,targets=pack_view(view,cfg.pad_token_id);tokens=tokens.to('neuron');targets=targets.to('neuron')
     if rank==0:(out/'progress.json').write_text(json.dumps(dict(arm=arm,split=split,batch=batch,wall_seconds=time.time()-started)))
     r=model.sparse_forward(tokens,cos,sin,anchors,raw_positions=where,mode='constant' if arm=='constant' else 'latent',pause_id=p['pause_token_id'],mix=.5)
     grid=readout_grid(r['raw_logits'],r['mixed_logits'],targets[:,127:159],selected,alphas=p['alphas'],temperatures=p['temperatures']);g={k:v.detach().cpu().tolist() for k,v in grid.items()}
     if split=='train':row=dict(index=index,domain=view['domain'],**g)
     else:
      ai,ti,rt=choice['alpha_index'],choice['temperature_index'],choice['raw_temperature_index'];row=dict(index=index,domain=view['domain'],policy32_nll=g['policy32'][ai][ti],workspace_anchor_nll=g['worked_anchor'][ai][ti],same_temperature_raw_anchor_nll=g['raw_anchor'][ti],calibrated_raw32_nll=g['raw32'][rt],calibrated_raw_anchor_nll=g['raw_anchor'][rt],fixed_raw32_nll=g['raw32'][0],fixed_raw_anchor_nll=g['raw_anchor'][0],fixed_workspace_anchor_nll=g['worked_anchor'][-1][0])
     rows.append(row)
    (out/f'rank{rank}-{arm}-{split}-scores.json').write_text(json.dumps(rows,indent=2)+'\n');dist.barrier()
    if split=='train':
     if rank==0:
      all_rows=[v for k in range(world) for v in json.loads((out/f'rank{k}-{arm}-train-scores.json').read_text())];choice=choose_readout(all_rows,alphas=p['alphas'],temperatures=p['temperatures'],ordinary=arm=='ordinary',split='train');choices[arm]=choice;(out/(arm+'-frozen-train-choice.json')).write_text(json.dumps(choice,indent=2)+'\n')
     else:choice=json.loads((out/(arm+'-frozen-train-choice.json')).read_text()) if (out/(arm+'-frozen-train-choice.json')).exists() else None
     dist.barrier();choice=json.loads((out/(arm+'-frozen-train-choice.json')).read_text())
    elif rank==0:
     all_rows=[v for k in range(world) for v in json.loads((out/f'rank{k}-{arm}-tune-scores.json').read_text())];summary[arm]={field:sum(w*sum(v[field] for v in all_rows if v['domain']==d)/sum(v['domain']==d for v in all_rows) for d,w in enumerate([.6,.2,.2])) for field in rows[0] if field.endswith('_nll')}
     expected=p['prior_scores'][arm];errors={n:abs(summary[arm]['fixed_'+n]-expected[n]) for n in ['raw32_nll','raw_anchor_nll','workspace_anchor_nll']};raw_parity[arm]=errors
     if max(errors.values())>1e-5:raise AssertionError('Frozen original endpoint readouts do not reproduce')
   if versions!=tuple(v._version for v in model.parameters()):raise AssertionError('Readout diagnostic changed model parameters')
 fallback=torch_neuronx.get_fallback_ops()
 if fallback:raise AssertionError('Readout fallback:'+str(fallback))
 if rank==0:(out/'report.json').write_text(json.dumps(dict(status='COMPLETE_SPLICED_GLOBAL_TRAIN_READOUT',world=world,optimizer_updates=0,calibration_views=p['calibration_views'],evaluation_anchors=p['evaluation_anchors'],choices=choices,scores=summary,original_readout_parity=raw_parity,parameters_unchanged=True,fallback_ops=fallback,wall_seconds=time.time()-started,scope='Fixed global mixer/temperature learned on TRAIN only,then frozen for reused TUNE.No per-token outcome-aware selection,backbone updates,allocation or independent confirmation.'),indent=2)+'\n')
 dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus');p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--comm-port',type=int,required=True);run(p.parse_args())
