import json
import os
from pathlib import Path
import socket

import torch
import torch.multiprocessing as mp
import pytest

from compute_trainium.scale_sharded_candidate import scale_inventory
from scripts.qualify_scale_sharding import worker
from compute_trainium.scale_sharded_fixture import relative


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        return str(sock.getsockname()[1])


def test_scale_inventory_matches_meta_and_cannot_be_legacy_baseline():
    rows=scale_inventory()
    assert [r['parameters'] for r in rows]==[955955264,953197056]
    assert rows[1]['parameters']<rows[0]['parameters']
    for row in rows:
        assert row['adamw_parameters']+row['muon_parameters']==row['parameters']
        assert row['master_gradient_moment_bytes']>12*row['parameters']


def test_nonfinite_and_small_families_cannot_hide_in_numerical_aggregate():
    with pytest.raises(FloatingPointError):
        relative({'x':torch.ones(2)},{'x':torch.tensor([1.,float('nan')])})
    result=relative({'large':torch.ones(1000),'small':torch.ones(1)*1e-4},
                    {'large':torch.ones(1000),'small':torch.ones(1)*2e-4})
    assert result['relative_l2']<1e-4
    assert result['largest_named_relative_l2']==1.


@pytest.mark.parametrize('execution',['materialized','streamed'])
def test_sharded_shared_tied_model_matches_complete_optimizer_and_fresh_process(tmp_path,monkeypatch,execution):
    monkeypatch.setenv('MASTER_ADDR','127.0.0.1')
    monkeypatch.setenv('MASTER_PORT',free_port())
    mp.spawn(worker,args=(4,'gloo',str(tmp_path/'warm'),'warm',None,None,execution),nprocs=4,join=True)
    monkeypatch.setenv('MASTER_PORT',free_port())
    mp.spawn(worker,args=(4,'gloo',str(tmp_path/'replay'),'replay',str(tmp_path/'warm'),None,execution),nprocs=4,join=True)
    report=json.loads((tmp_path/'warm/report.json').read_text())
    replay=json.loads((tmp_path/'replay/report.json').read_text())
    assert report['tied_parameter_alias_preserved']
    assert len(report['records'])==3
    assert replay['records'][0]['separate_process_replay_weights']['maximum_absolute_error']==0
    assert replay['records'][0]['separate_process_replay_moments']['maximum_absolute_error']==0
    assert max(r['gradient']['relative_l2'] for r in report['records'])<1e-5
    assert max(r['parameter_update']['relative_l2'] for r in report['records'])<1e-4
