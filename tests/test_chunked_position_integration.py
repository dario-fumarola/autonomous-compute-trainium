"""End-to-end software fixture only; no authentic source or research checkpoint."""
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from compute_trainium.chunked_persistent_workspace import ChunkedPersistentWorkspace
from compute_trainium.model import ModelConfig
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts import diagnose_chunked_positions as diagnostic


def test_complete_readout_reconstructs_scores_and_exercises_memory_path(tmp_path, monkeypatch):
    # Deliberately tiny random numerical model and repeated integer arrays.
    # The fake receipts below exist only under pytest's temporary directory.
    c=ModelConfig(dim=8,layers=1,heads=2,kv_heads=2,ffn_dim=12,vocab_size=17,max_seq=512)
    torch.set_num_threads(2);torch.manual_seed(922)
    model=ChunkedPersistentWorkspace(c).double().eval()
    tokens=np.tile((np.arange(513)%17).astype(np.uint16),(384,1))
    metadata=[dict(id='fixture-'+str(i),group='fixture-group-'+str(i//2),
                   domain=('web','math','code')[i//128]) for i in range(384)]
    evaluation=tmp_path/'numerical-fixture';evaluation.mkdir()
    np.save(evaluation/'broad-tokens.npy',tokens)
    write_state(evaluation/'PRIVATE-broad-metadata.json',metadata)
    records={};arms={};scores={};selection={}
    for family in diagnostic.FAMILIES:
        phase='fixture-'+family;folder=tmp_path/'runs'/phase;folder.mkdir(parents=True)
        checkpoint=folder/'PRIVATE-u1.pt';torch.save(dict(model=model.state_dict()),checkpoint)
        arms[family]=dict(phase=phase,checkpoint_updates=[1])
        records['audit-'+family]=dict(status='PASS_INTERLEAVED_SCREEN_MATURE_AUDITS',
            rows=[dict(checkpoint_sha256=file_sha256(checkpoint))])
        selection[family]=dict(arm=family)
        with torch.inference_mode():
            x=torch.from_numpy(tokens[:1,:-1].astype(np.int64))
            y=torch.from_numpy(tokens[:1,1:].astype(np.int64))
            logits=model.cached(x,128,int(family!='ordinary'),'pause' if family=='constant' else 'latent')
            loss=float(F.cross_entropy(logits.flatten(0,1),y.flatten()))
        scores[family]=dict(broad_nll=loss)
    main=dict(arms=arms,model_config=c.__dict__,evaluation_relative=evaluation.name,
        evaluation_files={f.name:file_sha256(f) for f in evaluation.iterdir()})
    root=tmp_path/diagnostic.SCREEN;root.mkdir()
    write_state(root/'plan.json',main)
    records['compare']=dict(selection=selection,scores=scores,both_development_gates_pass=True,
                           decisions={'software_fixture_only':True})
    write_state(root/'report.json',dict(status='COMPLETE_CHUNKED_PERSISTENT_SCREEN',records=records))
    plan=dict(main_plan_sha256=file_sha256(root/'plan.json'),source_sha256={},
        absolute_deadline_unix=time.time()+180,seconds_limit=180,workers=4,threads=2,
        bootstrap_seed=3,bootstrap_replicates=8,scope='NUMERICAL SOFTWARE FIXTURE ONLY')
    out=tmp_path/diagnostic.ID;out.mkdir();write_state(out/'plan.json',plan)
    public=tmp_path/'public-fixture';public.mkdir()
    monkeypatch.setattr(diagnostic,'PUBLIC',public)
    monkeypatch.setattr(diagnostic,'managed_running',lambda:[])
    # Worker launch remains real; no accelerator package or source data is used.
    # Four tiny workers exercise all twelve production integrity checks.
    diagnostic.run(tmp_path)
    report=json.loads((out/'report.json').read_text())
    assert report['status']=='COMPLETE_CHUNKED_POSITION_DIAGNOSTIC'
    assert report['assigned_chips']==report['optimizer_updates']==report['protected_groups_scored']==0
    assert report['memory_intervention']=='completed'
    assert len(report['comparisons'])==3
    assert report['comparisons']['authentic_over_blocked_memory']['strata']['before_first_slot']['broad']['gain']==0
    assert sum(len(r['checks']) for r in report['workers'])==12
