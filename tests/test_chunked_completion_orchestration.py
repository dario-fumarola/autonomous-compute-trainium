"""Host-only fixtures: no training, checkpoints, corpus or Native devices."""
import json
import threading

import pytest

from compute_trainium.training import file_sha256
import scripts.complete_chunked_controls as completion


@pytest.mark.parametrize('unstable_constant',[False,True])
def test_only_original_remaining_opportunities_then_audited_readout(tmp_path,monkeypatch,unstable_constant):
    root=tmp_path/'root';out=root/completion.ID;out.mkdir(parents=True)
    previous=root/completion.ORIGINAL;previous.mkdir()
    names=['ordinary-lr0','ordinary-lr1','ordinary-lr2','generated-lr0','generated-lr1']
    remaining=['constant-lr0','constant-lr1','constant-lr2']
    records={}
    for a in names:
        records[a]=dict(initial_model_digest='same')
        records['audit-'+a]=dict(status='PASS_INTERLEAVED_SCREEN_MATURE_AUDITS')
    original=previous/'report.json';original.write_text(json.dumps(dict(status='STOPPED',records=records)))
    arms={a:dict(phase='phase-'+a,family=a.split('-')[0],lr=[.0003,.001,.003][int(a[-1])]) for a in names+remaining}
    p=dict(original_report_sha256=file_sha256(original),remaining_original_opportunities=remaining,arms=arms,
        unavailable_arms={'generated-lr2':dict(reason='original failure')},original_gate_status='UNMET',
        root_comm_ports={**{arms[a]['phase']:123 for a in remaining},completion.ID+'-evaluation':456},
        evaluation_points=[dict(arm=a) for a in arms])
    (out/'plan.json').write_text(json.dumps(p))
    public=tmp_path/'public';public.mkdir();monkeypatch.setattr(completion,'PUBLIC',public)
    monkeypatch.setattr(completion,'managed_running',lambda:[])
    monkeypatch.setattr(completion,'nonfinite_failure',lambda root,phase:phase=='phase-constant-lr2')
    calls=[];executors=[]
    class FakeExecution:
        def __init__(self,root,plan,out):
            self.plan=json.loads(plan.read_text());self.sha='sha';self.failed=threading.Event()
            self.cleanup_errors=[];self.closed=False;executors.append(self)
        def check(self):assert not self.failed.is_set()
        def native(self,phase,module,args,status):
            calls.append(('native',phase))
            if unstable_constant and phase=='phase-constant-lr2':
                self.failed.set();raise RuntimeError('fixture nonfinite')
            return dict(status=status,initial_model_digest='same')
        def cpu(self,identity,module,args,path,status):
            calls.append(('cpu',identity));path.parent.mkdir(parents=True,exist_ok=True)
            result=dict(status=status);path.write_text(json.dumps(result));return result
        def close(self,path):self.closed=True
    monkeypatch.setattr(completion,'ScreenExecution',FakeExecution)
    completion.run(root)
    report=json.loads((out/'report.json').read_text())
    assert report['status']=='COMPLETE_CHUNKED_CONTROL_COMPLETION'
    assert file_sha256(original)==p['original_report_sha256']
    assert [name for kind,name in calls if kind=='native']==[arms[a]['phase'] for a in remaining]+[completion.ID+'-evaluation']
    assert all(e.closed for e in executors)
    readout=json.loads((out/'readout-plan.json').read_text())
    expected=set(names+remaining)-({'constant-lr2'} if unstable_constant else set())
    assert set(readout['arms'])==expected
    assert {x['arm'] for x in readout['evaluation_points']}==expected
    assert set(readout['root_comm_ports'])=={completion.ID+'-evaluation'}
    if unstable_constant:
        assert 'constant-lr2' in report['unavailable']
        assert not any(name==completion.ID+'-audit-constant-lr2' for kind,name in calls)
