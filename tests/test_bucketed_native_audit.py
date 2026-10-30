"""Numerical software fixtures only; no research sources or model results."""
import argparse
import copy
import json
from pathlib import Path

import pytest
import torch

from compute_trainium.interleaved_objective import reference_loss_sum
from compute_trainium.pretraining_optimizer import parameter_groups
from compute_trainium.tied_interleaved_ordinary import make_model
from compute_trainium.training import file_sha256


class NumericSourceFixture:
    def __init__(self,root):pass
    def batch(self,indices):
        x=(torch.tensor(indices)[:,None]*7+torch.arange(8)[None,:])%61+3
        y=(x*3)%61+3
        mask=(x%13)!=0
        return x,y,mask
    def target_count(self,start,count):return int(self.batch(list(range(start,start+count)))[2].sum())


def prepare(tmp_path,damage):
    parent=tmp_path/'runs'/'warm';parent.mkdir(parents=True)
    p=dict(model_config=dict(dim=16,layers=1,heads=2,kv_heads=2,ffn_dim=32,vocab_size=64,max_seq=8),
        untied_prediction_table=False,workspace_mode='ordinary',data_relative='fixture',warm_phase='warm',
        batch=2,lr=.0003,cpu_phase_seconds_limit=120)
    torch.manual_seed(8451);m=make_model(p);initial=copy.deepcopy(m.state_dict())
    torch.save(initial,parent/'PRIVATE-initial.pt')
    opt=torch.optim.AdamW(parameter_groups(m,m.embedding.weight),lr=p['lr'],betas=(.9,.95),eps=1e-8,foreach=False,fused=False)
    data=NumericSourceFixture(None)
    for update in (1,104):
        start=(update-1)*128;x,y,mask=data.batch(list(range(start,start+128)))
        opt.zero_grad(set_to_none=True);loss=reference_loss_sum(m,x,y,mask,0,mode='ordinary')/mask.sum();loss.backward()
        gradient=dict(gradients={n:v.grad.clone() for n,v in m.named_parameters()},loss=float(loss.detach()),update=update)
        for trial in (('original1','bucketed1') if update==1 else ('bucketed1',)):
            torch.save(gradient,parent/f'PRIVATE-{trial}-gradient-u{update}.pt')
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
        state=dict(model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()))
        if update==1:
            torch.save(state,parent/'PRIVATE-original1-u1.pt')
            if damage:
                bad=copy.deepcopy(state)
                for v in bad['optimizer']['state'].values():v['exp_avg'].mul_(.5)
                torch.save(bad,parent/'PRIVATE-bucketed1-u1.pt')
            else:torch.save(state,parent/'PRIVATE-bucketed1-u1.pt')
            torch.save(state,parent/'PRIVATE-bucketed1-u103.pt')
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(p))
    return argparse.Namespace(root=tmp_path,plan=plan,plan_sha=file_sha256(plan),output=tmp_path/'audit')


@pytest.mark.parametrize('damage',[False,True])
def test_complete_cpu_audit_accepts_parity_and_rejects_moment_corruption(tmp_path,monkeypatch,damage):
    import compute_trainium.probability_workspace_data as sources
    from scripts.audit_bucketed_native import main
    monkeypatch.setattr(sources,'ProbabilitySources',NumericSourceFixture)
    monkeypatch.setattr(torch,'set_num_threads',lambda n:None)
    monkeypatch.setattr(torch,'set_num_interop_threads',lambda n:None)
    a=prepare(tmp_path,damage)
    if damage:
        with pytest.raises(AssertionError,match='update/moment'):main(a)
    else:
        main(a)
        report=json.loads((a.output/'report.json').read_text())
        assert report['status']=='PASS_BUCKETED_NATIVE_COMPLETE_CPU_AUDIT'
        assert set(report['checks'])=={'1','104','native_first_update','native_first_gradient'}
