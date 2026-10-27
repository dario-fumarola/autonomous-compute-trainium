"""One fixed trained point, one full global gradient, zero optimizer updates."""
import argparse
import json
import os
from pathlib import Path
import time

if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]
os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--corpus');p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True)
    p.add_argument('--variant',choices=['scaled16','boundary32','full32'],required=True);p.add_argument('--comm-port',type=int,required=True)
    a=p.parse_args();started=time.time()
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from compute_trainium.training import file_sha256,PackedCorpus,atomic_checkpoint
    from compute_trainium.scale_model_sharding import shard_model_from_cpu,StreamingOwnerOptimizer
    from compute_trainium.scale_sharded_fixture import gather_named
    from compute_trainium.scale_precision_probe import probe_policy
    from compute_trainium.parcae_candidate import ParcaeCandidateConfig
    from compute_trainium.parcae_training_candidate import map_after_shared_validation
    from compute_trainium.metrics import real_token_losses
    from scripts.qualify_native import digest
    if file_sha256(a.plan)!=a.plan_sha:raise AssertionError('Frozen precision diagnosis changed')
    plan=json.loads(a.plan.read_text());code=Path(__file__).resolve().parents[1]
    if (plan['purpose']!='unchanged_scale_point_precision_diagnosis_only'
            or plan['phases'][a.output.name]['variant']!=a.variant
            or plan['root_comm_ports'][a.output.name]!=a.comm_port):raise ValueError('Exact zero-update variant required')
    if any(file_sha256(code/f)!=s for f,s in plan['worker_source_sha256'].items()):raise AssertionError('Frozen probe sources changed')
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
    import torch_neuronx
    torch.set_num_threads(2);torch.neuron.set_device(0);dist.init_process_group('neuron')
    rank,world=dist.get_rank(),dist.get_world_size()
    if world!=64:raise ValueError('Same64-core geometry required')
    a.output.mkdir(exist_ok=True);folder=a.root/plan['artifact_directory'];folder.mkdir(exist_ok=True)
    initializer=a.root/plan['point_file'];packing=a.root/plan['data']['packed']
    if rank==0:
        if file_sha256(initializer)!=plan['point_sha256']:raise AssertionError('Unchanged trained point changed')
        checked=PackedCorpus.from_mapped(packing,1024,23,plan['data']['source_sha256'])
        if checked.fingerprint!=plan['data']['fingerprint']:raise AssertionError('Source changed')
    dist.barrier();packed=map_after_shared_validation(packing,plan['data']['manifest_sha256'])
    saved=torch.load(initializer,map_location='cpu',mmap=True,weights_only=False)
    if saved['purpose']!='ineligible_unchanged_trained_point_probe' or saved['eligible_for_science'] is not False:
        raise ValueError('Ineligible fixed-point diagnostic weights required')
    config=ParcaeCandidateConfig(**plan['model']);kind,fp16,scale=probe_policy(a.variant)
    with torch.device('meta'):model=kind(config,recurrent=True,initialize=False,fp16_body=fp16)
    model=shard_model_from_cpu(model,init_device_mesh('neuron',(64,)),saved['model'],'neuron');del saved
    optimizer=StreamingOwnerOptimizer(model,fused_adamw=True)
    before=gather_named(model);before_digest=digest(before) if rank==0 else None;del before
    model.compile_stages('neuron')
    def objective(logits,targets,mask):return real_token_losses(logits,targets,mask).sum()
    loss=torch.compile(objective,backend='neuron',fullgraph=True,dynamic=False)
    cursor=plan['source_cursor'];target_count=sum(packed.counts(cursor,512));optimizer.zero_grad();nll=torch.zeros((),device='neuron')
    noise=torch.load(a.root/plan['noise_directory']/f'noise-u299-rank{rank}.pt',map_location='cpu',mmap=True,weights_only=False)
    if noise['windows']!=[cursor+i*64+rank for i in range(8)]:raise AssertionError('Frozen noise windows differ')
    for micro in range(8):
        window=cursor+micro*64+rank;batch=packed.batch([window],0)[0]
        inputs,targets,mask=[t.contiguous().to('neuron') for t in [batch.inputs,batch.targets,batch.target_mask]]
        state=noise['states'][micro:micro+1].contiguous().to('neuron')
        logits=model(inputs,state,total_steps=2,backward_cap=2);value=loss(logits,targets,mask)
        (value*(scale/target_count)).backward();nll.add_(value.detach())
    norm=optimizer.finish_gradients(scale=scale);dist.all_reduce(nll);gradient=gather_named(model,gradients=True)
    after=gather_named(model)
    if rank==0:
        if before_digest!=digest(after):raise AssertionError('Zero-update precision probe changed parameters')
        file=folder/(a.variant+'-gradient.pt');atomic_checkpoint(file,dict(gradients=gradient,loss=float(nll.cpu())/target_count,
            raw_gradient_norm=norm,real_targets=target_count,source_cursor=cursor,variant=a.variant,manual_loss_scale=scale))
        report=dict(status='CAPTURED_UNCHANGED_SCALE_PRECISION_GRADIENT',variant=a.variant,world=64,
            plan_sha256=a.plan_sha,optimizer_updates=0,parameters_unchanged=True,parameters=plan['parameters'],
            manual_loss_scale=scale,nll=float(nll.cpu())/target_count,raw_gradient_norm=norm,
            gradient_sha256=file_sha256(file),wall_seconds=time.time()-started,scientific_fit_admitted=False)
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
