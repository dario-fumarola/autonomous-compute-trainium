"""CPU counterfactual for the saved-output backward correction;not a device fix."""
import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA','0')
import torch

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.repro_sdpa_math2_cpu import comparison


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--sdk-source',type=Path,required=True);args=parser.parse_args();started=time.time()
    prepared=args.root/'sdpa-math2-cpu-v1';manifest=prepared/'manifest.json';info=json.loads(manifest.read_text())
    fixture=prepared/info['fixtures_file']
    if file_sha256(fixture)!=info['fixtures_sha256'] or file_sha256(args.sdk_source)!=info['sdk_python_source_sha256']:
        raise AssertionError('Exact original fixtures/source required')
    out=args.root/'sdpa-math2-correction-cpu-v1';public=Path('results/sdpa_math2_investigation_20261025/correction-cpu-v1')
    out.mkdir(exist_ok=False);public.mkdir(parents=True,exist_ok=False)
    plan={'registered_unix':started,'source_sha256':file_sha256(__file__),'manifest_sha256':file_sha256(manifest),
        'sdk_source_sha256':file_sha256(args.sdk_source),'seconds_limit':120,'research_examples':0,
        'scope':'Same8 frozen CPU numerical arrays. Substitute a recomputed unrounded output only in the installed Python backward to isolate its correction term. This is not a valid native operator-input substitution or a proposed deployed fix. Original SDK/fixtures unchanged.'}
    write_state(out/'plan.json',plan);write_state(public/'plan.json',plan)
    spec=importlib.util.spec_from_file_location('isolated_math2_correction',args.sdk_source)
    sdk=importlib.util.module_from_spec(spec);spec.loader.exec_module(sdk);torch.set_num_threads(4)
    rows=[]
    for case in torch.load(fixture,map_location='cpu',mmap=True,weights_only=False)['cases']:
        if time.time()-started>120:raise TimeoutError('Finite CPU correction diagnostic')
        q,k,v=[case[key].clone().requires_grad_() for key in ['q','k','v']]
        output,lse=sdk.compute_attention(q,k,v,None,None,case['scale'],True,0.)
        output.backward(case['grad_out'])
        truth={'output':output.detach(),'q':q.grad.detach(),'k':k.grad.detach(),'v':v.grad.detach()}
        q,k,v=q.detach(),k.detach(),v.detach()
        logits=sdk._apply_attention_mask(q.float()@k.float().transpose(-2,-1)*case['scale'],None,True)
        probabilities=torch.exp(logits-lse.detach()[...,None]);local_output=probabilities@v.float()
        interventions={}
        for label,saved_output in [('original_saved_output',output.detach()),('recomputed_unrounded_output',local_output)]:
            dq,dk,dv,_=sdk.compute_attention_backward(case['grad_out'],q,k,v,None,None,
                saved_output,lse.detach(),case['scale'],True,0.)
            interventions[label]=comparison(truth,{'output':output.detach(),'q':dq,'k':dk,'v':dv})
        rows.append({'fixture':case['identity'],'interventions':interventions})
    result={'status':'COMPLETE_CPU_CORRECTION_TERM_DIAGNOSTIC','records':rows,
        'wall_seconds':time.time()-started,'assigned_chips':0,'optimizer_updates':0,'research_examples':0,
        'test_accessed':False,'scope':plan['scope']}
    write_state(out/'report.json',result);write_state(public/'report.json',result)
    print(json.dumps({'status':result['status'],'wall_seconds':result['wall_seconds']}))


if __name__=='__main__':main()
