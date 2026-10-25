"""Numerical fixture for position/noise-preserving natural-context removal."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA','0')
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

from compute_trainium.code_context_views import views
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_training_candidate import initial_state
from compute_trainium.queue import write_state
from compute_trainium.reference_attention_mask_fixture import explicit_reference_mask
from compute_trainium.training import file_sha256
from scripts.audit_parcae_precise_training_point import make_reference
from scripts.qualify_native import digest
from scripts.qualify_parcae_candidate_cpu import PIN


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();started=time.time()
    out=a.root/'natural-code-context-v1';result_file=out/'intervention-qualification-v3.json'
    if result_file.exists():raise FileExistsError('No silent intervention qualification replacement')
    upstream=a.root/'upstream-parcae-69284c1'
    if subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=PIN or subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip():raise ValueError('Clean pinned upstream required')
    sys.path.insert(0,str(upstream));torch.set_num_threads(4);torch.manual_seed(137)
    selection=json.loads((out/'scoring-views-v3-report.json').read_text());config=json.loads((out/'plan.json').read_text())
    if file_sha256(out/'scoring-panel-v3.json')!=selection['panel_sha256']:raise AssertionError('Frozen target panel changed')
    point=json.loads((out/'scoring-panel-v3.json').read_text())['records'][0]
    corpus=a.root/'variable-depth-fresh-v1/dev.json'
    if file_sha256(corpus)!=config['source_sha256']:raise AssertionError('Original code corpus changed')
    document=next(d for d in json.loads(corpus.read_text()) if d['id']==point['document_id'])
    v=views(document['tokens'],point);small=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=128,vocab_size=32000)
    model=make_reference(small,True).eval()
    with torch.no_grad():
        for parameter in model.parameters():parameter.add_(torch.randn_like(parameter)*.01)
    before=digest(dict(sorted(model.state_dict().items())));state=initial_state(small,point['state_key'],1381)
    full=torch.tensor([v['full']['inputs']]);short=torch.tensor([v['short']['inputs']]);offset=v['short']['state_start']
    length=full.shape[1];mask=torch.zeros((length,length),dtype=torch.bool)
    mask[:offset,:offset]=torch.ones((offset,offset),dtype=torch.bool).tril()
    mask[offset:,offset:]=torch.ones((length-offset,length-offset),dtype=torch.bool).tril()
    changed=full.clone();changed[:,:offset]=(changed[:,:offset]+1)%32000
    current=None;model.initialize_state=lambda _:current;records=[]
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        for depth in [1,2]:
            kwargs={'num_steps_pair':torch.tensor([depth-1,1]),'return_logits':True}
            current=state[:,:length]
            # The pinned upstream mixer ignores the public attention_mask
            # argument. This isolated fixture injects the mask at SDPA instead.
            with explicit_reference_mask(mask):
                blocked=model(full,position_ids=torch.arange(length),**kwargs)['logits'][:,offset:]
                counterfactual=model(changed,position_ids=torch.arange(length),**kwargs)['logits'][:,offset:]
            unblocked=model(full,position_ids=torch.arange(length),**kwargs)['logits'][:,offset:]
            unblocked_changed=model(changed,position_ids=torch.arange(length),**kwargs)['logits'][:,offset:]
            current=state[:,offset:length]
            retained=model(short,position_ids=torch.tensor(v['short']['positions']),**kwargs)['logits']
            error=float((blocked-retained).abs().max());invariance=float((blocked-counterfactual).abs().max())
            sensitivity=float((unblocked-unblocked_changed).abs().max())
            print(json.dumps({'depth':depth,'short_error':error,'invariance_error':invariance,'sensitivity':sensitivity}),flush=True)
            if error>2e-5 or invariance>2e-5 or sensitivity<=1e-5:raise AssertionError('Position-preserving context fixture failed')
            records.append({'depth':depth,'masked_full_minus_position_preserved_short_max_abs':error,
                'removed_prefix_invariance_max_abs':invariance,'unmasked_prefix_change_sensitivity_max_abs':sensitivity})
    if digest(dict(sorted(model.state_dict().items())))!=before:raise AssertionError('Intervention rehearsal changed weights')
    if time.time()-started>300:raise TimeoutError('Context rehearsal five-minute cap')
    result={'status':'PASS_POSITION_NOISE_PRESERVING_CONTEXT_INTERVENTION','records':records,
        'panel_sha256':selection['panel_sha256'],'upstream_commit':PIN,'fixture_model_dim':32,
        'mask_adapter_scope':'Explicit CPU SDPA mask in numerical rehearsal only; unmasked scientific forwards use untouched upstream attention.',
        'source_sha256':{f:file_sha256(f) for f in ['scripts/qualify_code_context_intervention_v3.py','compute_trainium/code_context_views.py','compute_trainium/reference_attention_mask_fixture.py']},
        'assigned_chips':0,'optimizer_updates':0,'model_quality_metrics_reported':False,'test_accessed':False,
        'wall_seconds':time.time()-started,'scope':'Numerical rehearsal on tiny random perturbed upstream model. Counterfactual token arrays are software fixtures only and never research corpus labels. Exact retained source/position/hidden-state mapping; no scientific outcome.'}
    write_state(result_file,result);write_state(Path('results/natural_code_context_20261024/intervention-qualification-v3.json'),result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
