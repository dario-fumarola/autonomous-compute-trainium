"""Unchanged trainer with invalid-state capture before rethrowing a failed gradient."""
import json,sys,os
from pathlib import Path

if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]
os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'


def main():
    import torch
    import torch.distributed as dist
    from compute_trainium.scale_training_state import ScaleTrainingOptimizer
    from compute_trainium.scale_sharded_candidate import local_parameter
    from compute_trainium.scale_sharded_fixture import cpu_tree
    from compute_trainium.scale_model_state import replicated_state
    from compute_trainium.training import atomic_checkpoint,file_sha256
    from scripts.qualify_native import digest
    from scripts import train_scale_registered
    args=sys.argv[1:];root=Path(args[args.index('--root')+1]);plan_path=Path(args[args.index('--plan')+1]);case_key=args[args.index('--case')+1]
    plan=json.loads(plan_path.read_text());case=plan['cases'][case_key]
    if plan.get('invalid_gradient_capture') is not True:raise ValueError('Explicit registered failure preservation required')
    original=ScaleTrainingOptimizer.finish_gradients
    def checked(optimizer,*a,**kw):
        try:return original(optimizer,*a,**kw)
        except FloatingPointError as error:
            if str(error)!='Nonfinite global sharded gradient; no update':raise
            rank,world=dist.get_rank(),dist.get_world_size();folder=root/case['artifact_directory']/('invalid-u'+str(optimizer.update));folder.mkdir(parents=True,exist_ok=True)
            payload=cpu_tree(dict(purpose='scale_invalid_gradient_state_v1',eligible_for_science=False,
                signature=optimizer.lineage,rank=rank,world=world,completed_optimizer_updates=optimizer.update,
                failing_source_cursor=optimizer.update*512,loss_scale=kw.get('scale'),gradient_scope='unscaled,before clipping;no next optimizer update',
                model_shards={n:local_parameter(p) for n,p in optimizer.named},
                gradient_shards={n:local_parameter(p.grad) for n,p in optimizer.named},
                optimizer=optimizer.state_dict(),cpu_rng=torch.get_rng_state()))
            file=folder/f'rank{rank}.pt';atomic_checkpoint(file,payload)
            if digest(torch.load(file,map_location='cpu',mmap=True,weights_only=False))!=digest(payload):raise AssertionError('Invalid-state disk readback failed')
            note=dict(rank=rank,file=file.name,sha256=file_sha256(file),completed_optimizer_updates=optimizer.update,
                replicated_state_digest=digest(replicated_state(optimizer)))
            (folder/f'rank{rank}.json').write_text(json.dumps(note,indent=2)+'\n');dist.barrier()
            if rank==0:
                notes=[json.loads((folder/f'rank{i}.json').read_text()) for i in range(world)]
                (folder/'manifest.json').write_text(json.dumps(dict(status='CAPTURED_INVALID_GRADIENT_STATE_NOT_ELIGIBLE',
                    world=world,case=case_key,original_error=str(error),optimizer_updates_after_failure=0,
                    all_replicated_states_equal=len({n['replicated_state_digest'] for n in notes})==1,
                    all_shards_retained=True,no_majority_selection=True,eligible_for_science=False,shards=notes),indent=2)+'\n')
            dist.barrier();raise
    ScaleTrainingOptimizer.finish_gradients=checked
    try:train_scale_registered.main()
    finally:ScaleTrainingOptimizer.finish_gradients=original


if __name__=='__main__':main()
