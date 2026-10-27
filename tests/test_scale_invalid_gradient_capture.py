import json,sys
from types import SimpleNamespace

import pytest
import torch


def test_invalid_gradient_is_preserved_and_rethrown_without_an_update(tmp_path,monkeypatch):
    from scripts import train_scale_capture_invalid_gradient as driver
    from scripts import train_scale_registered
    from compute_trainium.scale_training_state import ScaleTrainingOptimizer
    from compute_trainium import scale_model_state,scale_sharded_candidate
    parameter=torch.nn.Parameter(torch.tensor([1.,2.]));parameter.grad=torch.tensor([float('nan'),float('nan')])
    signature=dict(phase='tune',study_sha256='software-fixture')
    optimizer=SimpleNamespace(named=[('weight',parameter)],update=2,lineage=signature,
        state_dict=lambda:dict(update=2,momentum=torch.tensor([3.,4.])))
    def fail(*args,**kwargs):raise FloatingPointError('Nonfinite global sharded gradient; no update')
    monkeypatch.setattr(ScaleTrainingOptimizer,'finish_gradients',fail)
    monkeypatch.setattr(torch.distributed,'get_rank',lambda:0)
    monkeypatch.setattr(torch.distributed,'get_world_size',lambda:1)
    monkeypatch.setattr(torch.distributed,'barrier',lambda:None)
    monkeypatch.setattr(scale_model_state,'replicated_state',lambda opt:{'weight':parameter.detach()})
    monkeypatch.setattr(scale_sharded_candidate,'local_parameter',lambda value:value)
    monkeypatch.setattr(train_scale_registered,'main',lambda:ScaleTrainingOptimizer.finish_gradients(optimizer,scale=65536))
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(dict(invalid_gradient_capture=True,cases={'fixture':{'artifact_directory':'artifacts'}})))
    monkeypatch.setattr(sys,'argv',['fixture','--root',str(tmp_path),'--plan',str(plan),'--case','fixture'])
    with pytest.raises(FloatingPointError,match='Nonfinite global'):driver.main()
    assert ScaleTrainingOptimizer.finish_gradients is fail and optimizer.update==2
    assert torch.equal(parameter.detach(),torch.tensor([1.,2.]))
    folder=tmp_path/'artifacts/invalid-u2';payload=torch.load(folder/'rank0.pt',weights_only=False)
    assert payload['eligible_for_science'] is False and payload['failing_source_cursor']==1024
    assert torch.isnan(payload['gradient_shards']['weight']).all()
    assert torch.equal(payload['optimizer']['momentum'],torch.tensor([3.,4.]))
    manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['all_shards_retained'] and manifest['no_majority_selection']
