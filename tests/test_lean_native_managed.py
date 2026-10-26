import copy
import json
import os
import sys
from pathlib import Path
import pytest
from scripts.qualify_lean_ntp_managed import validate_phase,main


def plan():
    return dict(purpose='lean_ntp_native_qualification_only',root_comm_ports={'one-warm':16001,'fixed-replay':16002},
                phases={'one-warm':dict(arm='fixed1',mode='warm'),
                        'fixed-replay':dict(arm='poisson2',mode='replay',parent='fixed-warm')})


def test_owned_port_cannot_be_borrowed_by_another_phase_or_model():
    p=plan();validate_phase(p,'one-warm',16001,arm='fixed1',mode='warm')
    with pytest.raises(ValueError,match='owned phase'):validate_phase(p,'one-warm',16002,arm='fixed1',mode='warm')
    with pytest.raises(ValueError,match='owned phase'):validate_phase(p,'unregistered',16001,arm='fixed1',mode='warm')
    with pytest.raises(ValueError,match='model or replay'):validate_phase(p,'one-warm',16001,arm='poisson2',mode='warm')
    with pytest.raises(ValueError,match='model or replay'):validate_phase(p,'fixed-replay',16002,arm='poisson2',mode='warm')


def test_scientific_purpose_cannot_enter_the_qualification_bootstrap():
    p=plan();p['purpose']='scientific_training'
    with pytest.raises(ValueError,match='Qualification-only'):validate_phase(p,'one-warm',16001,arm='fixed1',mode='warm')


@pytest.mark.parametrize('mode',['warm','replay'])
def test_owned_bootstrap_removes_only_its_arguments_and_preserves_legacy_inputs(tmp_path,monkeypatch,mode):
    from compute_trainium.training import file_sha256
    from scripts import qualify_lean_ntp_primitives as prepared
    preparation=tmp_path/'prepared';preparation.mkdir();(preparation/'plan.json').write_text('{}')
    execution=tmp_path/'execution';execution.mkdir();file=execution/'plan.json'
    label,arm,port=('one-warm','fixed1',16001) if mode=='warm' else ('fixed-replay','poisson2',16002)
    output=tmp_path/'runs'/label;output.mkdir(parents=True)
    sources={s:file_sha256(s) for s in ['scripts/qualify_lean_ntp_managed.py','scripts/train_parcae_precise_qualification.py']}
    p=plan();p.update(preparation_directory='prepared',preparation_plan_sha256=file_sha256(preparation/'plan.json'),
                      source_sha256=sources,worker_source_sha256=sources)
    file.write_text(json.dumps(p));observed=[]
    def fake_legacy():
        observed.extend(sys.argv[1:]);(output/'report.json').write_text('{}')
    monkeypatch.setattr(prepared,'main',fake_legacy);monkeypatch.setenv('RANK','0')
    monkeypatch.setenv('NEURON_RT_ROOT_COMM_ID','previous-test-value')
    argv=['worker','--qualification-plan',str(file),'--qualification-sha',file_sha256(file),
        '--comm-port',str(port),'--preparation-root',str(preparation),'--output',str(output),'--arm',arm,
        '--mode',mode,'--precision','fp16','--accumulation','16','--corpus','authentic-source-manifest']
    if mode=='replay':
        checkpoint=tmp_path/'runs/fixed-warm/after-120-full-state.pt';checkpoint.parent.mkdir()
        checkpoint.write_bytes(b'unit fixture,not research state')
        argv+=['--resume',str(checkpoint),'--resume-sha',file_sha256(checkpoint)]
    monkeypatch.setattr(sys,'argv',argv.copy())
    main()
    assert '--comm-port' not in observed and '--qualification-plan' not in observed
    assert observed[observed.index('--output')+1]==str(output)
    assert observed[observed.index('--corpus')+1]=='authentic-source-manifest'
    assert observed[observed.index('--arm')+1]==arm
    assert json.loads((output/'report.json').read_text())['managed_qualification_plan_sha256']==file_sha256(file)
    assert os.environ['NEURON_RT_ROOT_COMM_ID']=='127.0.0.1:'+str(port)
    if mode=='replay':
        assert observed[observed.index('--resume')+1]==str(checkpoint)
        assert '--resume-sha' not in observed
        observed.clear();argv[argv.index('--resume-sha')+1]='0'*64
        monkeypatch.setattr(sys,'argv',argv.copy())
        with pytest.raises(AssertionError,match='registered warm-state'):main()
        assert observed==[]
