import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pytest
from compute_trainium.future_span import FutureSpanConfig
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.training import file_sha256
from compute_trainium.future_span_scientific_state import INTEGRATION,signature
from scripts import run_future_span_primitive_driver as driver
from scripts.run_future_span_primitive_driver import register


def test_driver_cannot_register_or_claim_devices_before_hardware_pass(tmp_path):
    folder=tmp_path/'future-span-math-qualification-v2';folder.mkdir()
    (folder/'report.json').write_text(json.dumps({'status':'STOPPED_FUTURE_SPAN_QUALIFICATION'}))
    with pytest.raises(ValueError,match='hardware qualification'):
        register(tmp_path)
    assert not (tmp_path/'future-span-primitive-driver-v2').exists()


def test_four_cell_driver_registration_builds_valid_stable_prefix_branches(tmp_path,monkeypatch):
    # Numerical metadata fixture only;no source corpus,model updates or devices.
    public=tmp_path/'public'
    monkeypatch.setattr(driver,'Path',lambda value:public if value=='results/future_span_driver_20261025/primitive-v2' else Path(value))
    q=tmp_path/'future-span-math-qualification-v2';q.mkdir();prep=tmp_path/'prep';prep.mkdir()
    packing=tmp_path/'packing';packing.mkdir()
    cfg=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,vocab_size=64)
    span=FutureSpanConfig();windows=41*512
    counts=np.stack([np.arange(windows+1,dtype=np.int64)]*3,-1)
    np.save(packing/'prefix_counts.npy',counts)
    metadata={'files':{'prefix_counts.npy':file_sha256(packing/'prefix_counts.npy')}}
    (packing/'manifest.json').write_text(json.dumps(metadata))
    np.save(prep/'prefix.npy',np.arange(windows+1,dtype=np.int64)*63)
    tables={}
    for arm,n in [('one',1),('recurrent',2)]:
        path=prep/(arm+'.npy');np.save(path,np.full((121,16,32,2),n,dtype=np.uint16))
        tables[arm]={'file':path.name,'sha256':file_sha256(path)}
    manifest={'model':asdict(cfg),'future_config':asdict(span),'data':{'packed':'packing',
        'manifest_sha256':file_sha256(packing/'manifest.json'),'packing_fingerprint':'fixture-packing',
        'corpus_sha256':'fixture-source'},'depths':tables,'prefix_file':'prefix.npy',
        'prefix_sha256':file_sha256(prep/'prefix.npy'),'backbone_model_seed':1009,'state_seed':2009,
        'training_parameters':cfg.parameter_count(True)+cfg.dim*cfg.vocab_size+cfg.vocab_size,
        'initial_file':'initial.pt','initial_sha256':'fixture-initial','model_digest':'fixture-model',
        'head_seed':104729,'routing_seed':271828}
    (prep/'manifest.json').write_text(json.dumps(manifest))
    cells={a+'-'+label:{'identity':a+'-'+label,'arm':a,'objective_weight':weight}
           for a in ['one','recurrent'] for label,weight in [('ntp',0.),('future',.1)]}
    plan={'preparation':'prep','preparation_manifest_sha256':file_sha256(prep/'manifest.json'),
        'source_sha256':{},'cases':cells}
    (q/'plan.json').write_text(json.dumps(plan))
    (q/'report.json').write_text(json.dumps({'status':'PASS_COMPLETE_FUTURE_SPAN_TRAINING_QUALIFICATION',
        'plan_sha256':file_sha256(q/'plan.json')}))
    _,_,registered=register(tmp_path)
    assert registered['purpose']==INTEGRATION and len(registered['cases'])==8
    assert len(registered['root_comm_ports'])==16
    for cell in cells:
        parent=registered['cases'][cell+'-parent'];child=registered['cases'][cell+'-branch']
        assert parent['warmup_targets']<parent['cooldown_start_targets']<parent['end_targets']
        assert child['parent']=={'case_id':cell+'-parent','update':20}
        assert signature('same-plan',registered,parent)==signature('same-plan',registered,child)
