"""New lineage/checkpoint wrapper fixture; numerical arrays, no research data."""
import argparse
import json
import os
from pathlib import Path
import time

if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from compute_trainium.scale_sharded_fixture import SoftwareModel,loss,gather_named,full_moments,relative
from compute_trainium.scale_model_sharding import shard_model_from_cpu
from compute_trainium.scale_training_state import ScaleTrainingOptimizer,save_training_shard,load_training_shard,verify_live_training_state

class CounterFixture:
    def counts(self,first,number):return [number*3,number,number]


def state_worker(rank,world,folder,mode,backend='gloo',parent=None,comm_port=None):
    torch.set_num_threads(1)
    if backend=='neuron':
        os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(comm_port)
        import torch_neuronx
        torch.neuron.set_device(0);device='neuron'
    else:device='cpu'
    dist.init_process_group(backend,rank=rank,world_size=world)
    torch.manual_seed(3719);model=SoftwareModel();state={n:t.clone() for n,t in model.state_dict().items()}
    with torch.device('meta'):model=SoftwareModel()
    model=shard_model_from_cpu(model,init_device_mesh(device,(world,)),state,device)
    signature=dict(study_sha256='fixture',case='fixed-two',phase='gate',initializer_sha256='unupdated',
        packing_fingerprint='software-only',model_role='fixed_two',state_seed=7)
    optimizer=ScaleTrainingOptimizer(model,lineage=signature)
    folder=Path(folder);counter=CounterFixture();folder.mkdir(exist_ok=True,parents=True);parent=Path(parent) if parent else folder
    if mode=='replay':
        start=load_training_shard(model,optimizer,parent/'warm',signature,1,counter,10**15)
    else:start=0
    for update in range(start,2):
        optimizer.zero_grad();value=loss(model,rank,world,update+1,device);value.backward()
        optimizer.finish_gradients();optimizer.step()
        save_training_shard(model,optimizer,folder/mode,signature,update+1,counter,10**15)
    # A diagnostic backward must leave all serialized state unchanged. The
    # checker must detect a mutation rather than silently restoring over it.
    optimizer.zero_grad();loss(model,rank,world,3,device).backward();optimizer.finish_gradients()
    verify_live_training_state(model,optimizer,folder/mode,signature,2,10**15)
    first=next(model.parameters());local=first.to_local();original=local.detach().clone()
    with torch.no_grad():local.add_(.01)
    detected=False
    try:verify_live_training_state(model,optimizer,folder/mode,signature,2,10**15)
    except AssertionError:detected=True
    if not detected:raise AssertionError('A deliberate local parameter mutation was not detected')
    with torch.no_grad():local.copy_(original)
    weights=gather_named(model);moments=full_moments(optimizer)
    if rank==0:
        if mode=='warm':torch.save(dict(weights=weights,moments=moments),folder/'expected.pt')
        else:
            expected=torch.load(parent/'expected.pt',weights_only=False)
            result=dict(weights=relative(expected['weights'],weights),moments=relative(expected['moments'],moments))
            assert result['weights']['maximum_absolute_error']==0
            assert result['moments']['maximum_absolute_error']==0
            (folder/'result.json').write_text(json.dumps(result))
    dist.barrier();dist.destroy_process_group()



def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--corpus');p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True)
    p.add_argument('--mode',choices=['warm','replay'],required=True);p.add_argument('--parent',type=Path)
    p.add_argument('--comm-port',type=int,required=True);a=p.parse_args();started=time.time()
    from compute_trainium.training import file_sha256
    plan=json.loads(a.plan.read_text());source=Path(__file__).resolve().parents[1]
    if (file_sha256(a.plan)!=a.plan_sha or plan['purpose']!='scale_training_state_software_fixture_only'
            or plan['phases'][a.output.name]['mode']!=a.mode or plan['root_comm_ports'][a.output.name]!=a.comm_port):
        raise ValueError('Exact small state-fixture registration required')
    if any(file_sha256(source/f)!=sha for f,sha in plan['worker_source_sha256'].items()):raise AssertionError('Frozen source changed')
    state_worker(int(os.environ['RANK']),64,str(a.output),a.mode,'neuron',str(a.parent) if a.parent else None,a.comm_port)
    if int(os.environ['RANK'])==0:
        report=dict(status='PASS_SCALE_TRAINING_STATE_SOFTWARE_FIXTURE',world=64,mode=a.mode,
            plan_sha256=a.plan_sha,research_examples=0,research_optimizer_updates=0,
            actual_optimizer_updates=2 if a.mode=='warm' else 1,wall_seconds=time.time()-started)
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
