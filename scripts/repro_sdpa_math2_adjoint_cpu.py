"""Compare the installed explicit backward with autograd of its own forward.

Uses only the already frozen numerical operator fixtures. This distinguishes
different forward approximations from a discrepancy in the backward formula.
"""
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
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--sdk-source',type=Path,required=True);a=p.parse_args();started=time.time()
    prepared=a.root/'sdpa-math2-cpu-v1';info=json.loads((prepared/'manifest.json').read_text())
    if (file_sha256(a.sdk_source)!=info['sdk_python_source_sha256']
            or file_sha256(prepared/info['fixtures_file'])!=info['fixtures_sha256']):raise AssertionError('Frozen numerical fixtures/SDK changed')
    out=a.root/'sdpa-math2-adjoint-cpu-v1';public=Path('results/sdpa_math2_investigation_20261025/adjoint-cpu-v1')
    out.mkdir(exist_ok=False);public.mkdir(parents=True,exist_ok=False)
    registration={'identity':'sdpa-math2-adjoint-cpu-v1','registered_unix':started,
        'source_sha256':file_sha256(__file__),'parent_fixture_manifest_sha256':file_sha256(prepared/'manifest.json'),
        'sdk_source_sha256':info['sdk_python_source_sha256'],'seconds_limit':120,'assigned_chips':0,
        'optimizer_updates':0,'research_examples':0,'test_accessed':False,
        'scope':'CPU software arrays only,same8 frozen fixtures. Compare explicit SDK backward against PyTorch autograd through the identical SDK forward. No model,corpus,native execution,training or threshold change.'}
    write_state(out/'plan.json',registration);write_state(public/'plan.json',registration)
    result={'status':'STOPPED_CPU_SAME_FORWARD_ADJOINT_CHECK'};records=[]
    try:
        torch.set_num_threads(4)
        spec=importlib.util.spec_from_file_location('isolated_sdpa_math_adjoint',a.sdk_source)
        sdk=importlib.util.module_from_spec(spec);spec.loader.exec_module(sdk)
        cases=torch.load(prepared/info['fixtures_file'],map_location='cpu',mmap=True,weights_only=False)['cases']
        for case in cases:
            if time.time()-started>registration['seconds_limit']:raise TimeoutError('CPU adjoint fixture cap')
            q,k,v=[case[n].clone().requires_grad_() for n in ['q','k','v']]
            output,lse=sdk.compute_attention(q,k,v,None,None,case['scale'],True,0.)
            output.backward(case['grad_out'])
            expected={'output':output.detach(),'q':q.grad.detach(),'k':k.grad.detach(),'v':v.grad.detach()}
            dq,dk,dv,_=sdk.compute_attention_backward(case['grad_out'],q.detach(),k.detach(),v.detach(),
                None,None,output.detach(),lse.detach(),case['scale'],True,0.)
            actual={'output':output.detach(),'q':dq,'k':dk,'v':dv}
            records.append({'fixture':case['identity'],'explicit_backward_vs_own_forward_autograd':comparison(expected,actual)})
        if file_sha256(a.sdk_source)!=registration['sdk_source_sha256']:raise AssertionError('SDK source changed')
        result={'status':'COMPLETE_CPU_SAME_FORWARD_ADJOINT_CHECK','records':records,'fixture_cases':len(cases)}
    except BaseException as error:result['error']=str(error);raise
    finally:
        result.update(records=records,wall_seconds=time.time()-started,assigned_chips=0,optimizer_updates=0,
            research_examples=0,test_accessed=False,scope=registration['scope'])
        write_state(out/'report.json',result);write_state(public/'report.json',result)
        print(json.dumps({'status':result['status'],'wall_seconds':result['wall_seconds']}),flush=True)


if __name__=='__main__':main()
