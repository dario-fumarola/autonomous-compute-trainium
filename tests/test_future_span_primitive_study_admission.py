import json
from pathlib import Path
import pytest

from scripts import register_future_span_primitive_study as admission
from scripts.register_future_span_primitive_study import register
from compute_trainium.future_span_study_plan import CELLS,RATES,select_rates
from compute_trainium.training import file_sha256


def test_scientific_registration_rejects_missing_or_failed_gates_without_creating_run(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):register(tmp_path,'tune')
    q=tmp_path/'future-span-math-qualification-v2';q.mkdir()
    (q/'plan.json').write_text('{}')
    (q/'report.json').write_text(json.dumps({'status':'STOPPED'}))
    with pytest.raises(ValueError):register(tmp_path,'tune')
    assert not (tmp_path/'future-span-tune-v2').exists()
    assert not (tmp_path/'results').exists()


def test_tuning_then_main_metadata_preserve_fresh_seed_pairing_grids_and_own_cooldown(tmp_path,monkeypatch):
    # Numerical registration metadata only,no model,corpus or device work.
    public=tmp_path/'public'
    monkeypatch.setattr(admission,'Path',lambda p:public/str(p).replace('/','__') if str(p).startswith('results/') else Path(p))
    measured={'status':'PASS_COMPLETE_FUTURE_SPAN_TRAINING_QUALIFICATION','records':[
        {'case':{'identity':cell},'native':{'warm':{'warm_updates':100,'warm_seconds':80.,
            'wall_seconds':180.,'warm_targets_per_second':650000.}}} for cell in CELLS]}
    hardware=tmp_path/'future-span-math-qualification-v2';hardware.mkdir()
    (hardware/'report.json').write_text(json.dumps(measured))
    source=tmp_path/'variable-depth-fresh-v1';source.mkdir();(source/'dev.json').write_text('[]')
    initials={f's{seed}-family':{'seed':seed,'file':f'initial-s{seed}.pt','sha256':'unloaded-fixture',
        'head_seed':seed+104729,'training_parameters':166571840,'deployed_parameters':141963840}
        for seed in [4127,5231,7283]}
    tables={arm:{'depth_table':arm+'-depths.npy','depth_table_sha256':'fixture-depth',
        'work_table':arm+'-work.npy','work_table_sha256':'fixture-work','milestones':{
            key:{'updates':u,'real_targets':u*500000,'valid_anchors':u*30000,'estimated_dense_matrix_ops':u*1000000,
                'cooldown_start_update':u//2,'cooldown_start_targets':u*250000}
            for key,u in [('tuning',576),('main',1920)]}} for arm in ['one','recurrent']}
    manifest={'initial_states':initials,'tables':tables,'model':{},'future_config':{},'routing_seed':3,
        'data':{},'auxiliary_data':{}}
    monkeypatch.setattr(admission,'prerequisites',lambda root:({}, {}, manifest))
    _,_,tune=register(tmp_path,'tune')
    assert tune['execution_variant']=='fp16_body_fp32_attention_primitives'
    assert all(not p.startswith('third_party/') for p in tune['worker_source_sha256'])
    assert len(tune['cases'])==12 and len(tune['root_comm_ports'])==24
    for cell in CELLS:
        cases=[c for c in tune['cases'].values() if c['cell']==cell]
        assert {c['peak_lr'] for c in cases}==set(RATES) and {c['seed'] for c in cases}=={4127}
        assert all(c['cooldown_start_update']==288 and c['total_updates']==576 for c in cases)
    scores=[{'cell':case['cell'],'case_id':key,'peak_lr':case['peak_lr'],
             'metrics':{'weighted_nll':2.5+abs(case['peak_lr']-.016)}}
            for key,case in tune['cases'].items()]
    selected=select_rates(scores)
    report={'status':'COMPLETE_BOUNDED_FUTURE_SPAN_TUNING','plan_sha256':file_sha256(tmp_path/'future-span-tune-v2/plan.json'),
        'selected':selected,'tuning_scores':scores,
        'selected_gradient_gates':[{'case_id':row['case_id'],'independent_cpu':{
            'case_id':row['case_id'],'status':'PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT'}}
            for row in selected.values()]}
    (tmp_path/'future-span-tune-v2/report.json').write_text(json.dumps(report))
    gates=report['selected_gradient_gates']
    report['selected_gradient_gates']=[gates[0]]*4
    (tmp_path/'future-span-tune-v2/report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='checkpoints'):
        register(tmp_path,'main')
    assert not (tmp_path/'future-span-main-v2').exists()
    report['selected_gradient_gates']=gates
    (tmp_path/'future-span-tune-v2/report.json').write_text(json.dumps(report))
    _,_,main=register(tmp_path,'main')
    assert len(main['cases'])==8 and len(main['root_comm_ports'])==24
    assert {c['seed'] for c in main['cases'].values()}=={5231,7283}
    assert all(c['initial_state']==f's{c["seed"]}-family' and c['segment_ends']==[960,1920]
               and c['cooldown_start_update']==960 and c['gradient_updates']==[1920] for c in main['cases'].values())
    assert main['selected_rates']==selected and main['terminal_alpha_spent']==0
