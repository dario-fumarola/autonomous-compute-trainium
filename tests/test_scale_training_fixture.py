import json
import os
from pathlib import Path
import socket

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
import torch.multiprocessing as mp

from compute_trainium.scale_sharded_fixture import SoftwareModel,loss,gather_named,full_moments,relative
from compute_trainium.scale_model_sharding import shard_model_from_cpu
from compute_trainium.scale_training_state import ScaleTrainingOptimizer,save_training_shard,load_training_shard


from scripts.qualify_scale_training_state import state_worker


def port():
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));return str(sock.getsockname()[1])


def test_training_wrapper_retains_exact_fresh_process_state_and_updates(tmp_path,monkeypatch):
    monkeypatch.setenv('MASTER_ADDR','127.0.0.1')
    for mode in ['warm','replay']:
        monkeypatch.setenv('MASTER_PORT',port())
        mp.spawn(state_worker,args=(4,str(tmp_path),mode),nprocs=4,join=True)
    assert json.loads((tmp_path/'result.json').read_text())['weights']['relative_l2']==0
