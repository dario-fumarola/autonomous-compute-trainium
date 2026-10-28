"""Cross preserved prefix-only writers/readers without resampling or fitting."""
import argparse,json,os,time
from pathlib import Path
if 'RESEARCH_VISIBLE_CORES' in os.environ:os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def run(a):
 import torch
 import torch.distributed as dist
 from torch.distributed.device_mesh import init_device_mesh
 from compute_trainium.training import file_sha256
 from compute_trainium.workspace_olmo_candidate import WorkspaceOlmoCandidate,OlmoWorkspaceConfig
 from compute_trainium.workspace_native_state import restore_model_shards
 from compute_trainium.workspace_native_objective import SelectiveWorkspacePredictor,drain_runtime
 from compute_trainium.workspace_view_batches import WorkspaceViewBatches
 from compute_trainium.scale_model_sharding import shard_model_from_cpu
 p=json.loads(a.plan.read_text());source=Path(__file__).resolve().parents[1]
 if file_sha256(a.plan)!=a.plan_sha or p['purpose']!='workspace_writer_reader_crossover':raise AssertionError('Frozen crossover required')
 if any(file_sha256(source/n)!=h for n,h in p['worker_source_sha256'].items()):raise AssertionError('Source changed')
 rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']);torch.set_num_threads(2);started=time.time()
 os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
 import torch_neuronx
 torch.neuron.set_device(0);dist.init_process_group('neuron',rank=rank,world_size=world);mesh=init_device_mesh('neuron',(world,))
 out=a.output;out.mkdir(parents=True,exist_ok=True)
 note_file=a.root/'runs'/p['saved_note_run']/f'rank{rank}-PRIVATE-generated-notes.pt'
 if file_sha256(note_file)!=p['saved_note_sha256'][str(rank)]:raise AssertionError('Saved prefix-only draws changed')
 notes=torch.load(note_file,weights_only=True);lookup={(x['arm'],x['index']):x['workspace'] for x in notes if x['style']=='sample' and x['arm'] in ['c0','workspace']}
 c0=torch.load(a.root/p['checkpoint_relative'],weights_only=True,mmap=True)['model'];cfg=OlmoWorkspaceConfig(**p['model_config'])
 with torch.device('meta'):model=WorkspaceOlmoCandidate(cfg).eval()
 model=shard_model_from_cpu(model,mesh,c0,'neuron');del c0
 model.requires_grad_(False);model.compile_stages('neuron');actor=SelectiveWorkspacePredictor(model)
 data=WorkspaceViewBatches(a.root/p['data_relative'],'tune');all_results={};parity=0.
 with torch.no_grad():
  for reader in ['c0','workspace']:
   if reader=='workspace':restore_model_shards(a.root/'runs'/p['arm_runs'][reader]/f'rank{rank}-u{p["updates"][reader]}.zip',model,label='model')
   old={r['index']:r for r in json.loads((a.root/'runs'/p['original_evaluation_run']/f'rank{rank}-{reader}-losses.json').read_text())};rows=[]
   for batch in range(5):
    index=batch*world+rank;view=data.at(index);c,y=view['prefix'],view['continuation']
    if rank==0:(out/'progress.json').write_text(json.dumps(dict(reader=reader,batch=batch,wall_seconds=time.time()-started)))
    lp=actor.log_probs(c,y[:,:4],'crossover_raw').detach().cpu();drain_runtime(actor.device)
    row=dict(index=index,domain=view['domain'],raw_nll=float(-lp.mean()),raw_first_token_nll=float(-lp[:,0].mean()))
    parity=max(parity,abs(row['raw_nll']-old[index]['raw_first4_nll']))
    for writer in ['c0','workspace']:
     context=torch.cat([lookup[(writer,index)],c],1)
     lp=actor.log_probs(context,y[:,:4],'writer_'+writer).detach().cpu();drain_runtime(actor.device)
     row[writer+'_nll']=float(-lp.mean());row[writer+'_first_token_nll']=float(-lp[:,0].mean())
     if writer==reader:parity=max(parity,abs(row[writer+'_nll']-old[index]['own_nll']))
    if parity>1e-5:raise AssertionError('Original diagonal or raw output changed')
    rows.append(row)
   (out/f'rank{rank}-{reader}-losses.json').write_text(json.dumps(rows,indent=2)+'\n');dist.barrier()
   if rank==0:
    records=[x for k in range(world) for x in json.loads((out/f'rank{k}-{reader}-losses.json').read_text())]
    all_results[reader]={field:sum(w*sum(x[field] for x in records if x['domain']==d)/sum(x['domain']==d for x in records) for d,w in enumerate([.6,.2,.2])) for field in [k for k in rows[0] if k.endswith('_nll')]}
 fallback=torch_neuronx.get_fallback_ops()
 if fallback:raise AssertionError('Crossover used fallback')
 errors=torch.zeros(world);errors[rank]=parity;errors=errors.to('neuron');dist.all_reduce(errors);parity=float(errors.cpu().max())
 if rank==0:(out/'report.json').write_text(json.dumps(dict(status='COMPLETE_WORKSPACE_WRITER_READER_CROSSOVER',world=world,scores_by_reader=all_results,original_diagonal_max_abs_error=parity,wall_seconds=time.time()-started,fallback_ops=fallback,optimizer_updates=0,scope='Same320 reused TUNE anchors;saved prefix-only G1 draws,no regeneration or selection by answer.Off-diagonals separate writer/reader changes but do not isolate PG without a PG=0 training control.No fresh confirmation.')))
 dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus');p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--comm-port',type=int,required=True);run(p.parse_args())
