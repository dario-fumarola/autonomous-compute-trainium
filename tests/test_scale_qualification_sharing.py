import json
from pathlib import Path
import socket

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
import torch.multiprocessing as mp

from compute_trainium.scale_sharded_fixture import SoftwareModel,loss
from compute_trainium.scale_model_sharding import shard_model_from_cpu,StreamingOwnerOptimizer
from compute_trainium.scale_model_state import save_shard,load_shard,share_exact_replay_shard


class Counts:
    def counts(self,first,number):return [number*3,number,number]


def worker(rank,world,folder,mode,mutate):
    torch.set_num_threads(1);dist.init_process_group('gloo',rank=rank,world_size=world)
    torch.manual_seed(629);source=SoftwareModel();state=source.state_dict()
    with torch.device('meta'):model=SoftwareModel()
    model=shard_model_from_cpu(model,init_device_mesh('cpu',(world,)),state,'cpu')
    optimizer=StreamingOwnerOptimizer(model);folder=Path(folder);(folder/mode).mkdir(exist_ok=True,parents=True)
    signature={'case':'software-only'};counts=Counts();start=0
    if mode=='replay':load_shard(model,optimizer,folder/'warm',signature,1,counts);start=1
    for step in range(start,2):
        optimizer.zero_grad();loss(model,rank,world,step+1,'cpu').backward();optimizer.finish_gradients();optimizer.step()
        if mode=='warm':save_shard(model,optimizer,folder/mode,signature,step+1,counts,200*(step+1),10*(step+1))
    if mode=='replay':
        if mutate and rank==0:
            with torch.no_grad():next(model.parameters()).to_local().add_(.01)
        share_exact_replay_shard(model,optimizer,folder/mode,folder/'warm',signature,2,counts,400,20)
        assert (folder/mode/f'rank{rank}-u2.pt').is_symlink() is (not mutate)
    dist.barrier();dist.destroy_process_group()


def port():
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));return str(sock.getsockname()[1])


@pytest.mark.parametrize('mutate',[False,True])
def test_only_exact_live_replays_share_bytes_and_a_single_difference_keeps_every_raw_rank(tmp_path,monkeypatch,mutate):
    monkeypatch.setenv('MASTER_ADDR','127.0.0.1')
    for mode in ['warm','replay']:
        monkeypatch.setenv('MASTER_PORT',port());mp.spawn(worker,args=(4,str(tmp_path),mode,mutate),nprocs=4,join=True)
    manifest=json.loads((tmp_path/'replay/manifest-u2.json').read_text())
    assert manifest.get('all_live_replay_states_exact',False) is (not mutate)
    assert len(manifest['shards'])==4
