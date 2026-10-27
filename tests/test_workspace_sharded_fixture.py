import json
import socket
import torch.multiprocessing as mp
from scripts.qualify_workspace_sharded_fixture import worker


def free_port():
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));return str(sock.getsockname()[1])


def test_full_sharded_olmo_adamw_ema_and_fresh_process_replay(tmp_path,monkeypatch):
    monkeypatch.setenv('MASTER_ADDR','127.0.0.1');monkeypatch.setenv('MASTER_PORT',free_port())
    mp.spawn(worker,args=(4,'gloo',str(tmp_path/'warm'),'warm'),nprocs=4,join=True)
    monkeypatch.setenv('MASTER_PORT',free_port())
    mp.spawn(worker,args=(4,'gloo',str(tmp_path/'replay'),'replay',str(tmp_path/'warm')),nprocs=4,join=True)
    a=json.loads((tmp_path/'warm/report.json').read_text());b=json.loads((tmp_path/'replay/report.json').read_text())
    assert max(r['gradient']['relative_l2'] for r in a['records'])<1e-5
    assert b['records'][0]['fresh_process_exact_replay']
