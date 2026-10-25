"""Numerical-array reproduction of installed Math2 SDPA backward rounding.

These are software operator fixtures,not synthetic training/evaluation data.
No SDK file is modified;only its standalone Torch formulas are imported.
"""
import argparse
import importlib.util
import json
import math
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA','0')
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel

from compute_trainium.parcae_update_qualification import relative_error
from compute_trainium.queue import write_state
from compute_trainium.training import atomic_checkpoint,file_sha256
from scripts.qualify_native import digest


def fixture_cases():
    cases=[]
    for n in [32,1024]:
        for dtype in [torch.float16,torch.float32]:
            for same in [False,True]:
                g=torch.Generator().manual_seed(109)
                q=F.rms_norm(torch.randn(1,6,n,128,generator=g),(128,)).to(dtype)
                k=q.clone() if same else F.rms_norm(torch.randn(q.shape,generator=g),(128,)).to(dtype)
                v=(torch.randn(q.shape,generator=g)*.5).to(dtype);grad=torch.randn(q.shape,generator=g).to(dtype)
                cases.append({'identity':f'n{n}-{str(dtype).split(".")[-1]}-'+('self' if same else 'independent'),
                    'q':q,'k':k,'v':v,'grad_out':grad,'is_causal':True,'scale':1/math.sqrt(128)})
    return cases


def pytorch_reference(case):
    q,k,v=[case[n].clone().requires_grad_() for n in ['q','k','v']]
    with sdpa_kernel(SDPBackend.MATH):
        out=F.scaled_dot_product_attention(q,k,v,is_causal=True);out.backward(case['grad_out'])
    return {'output':out.detach(),'q':q.grad.detach(),'k':k.grad.detach(),'v':v.grad.detach()}


def comparison(expected,actual):
    return {key:relative_error({key:expected[key].float()},{key:actual[key].float()}) for key in expected}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--sdk-source',type=Path,required=True);a=p.parse_args();started=time.time()
    out=a.root/'sdpa-math2-cpu-v1';public=Path('results/sdpa_math2_investigation_20261025/cpu-v1')
    out.mkdir(exist_ok=False);public.mkdir(parents=True,exist_ok=False)
    plan={'identity':'sdpa-math2-cpu-v1','registered_unix':started,'seconds_limit':180,
        'source_sha256':{'scripts/repro_sdpa_math2_cpu.py':file_sha256(__file__)},
        'sdk_python_module':'torch_neuronx/_sdpa_math.py','sdk_python_source_sha256':file_sha256(a.sdk_source),
        'fixture_rule':{'seed':109,'heads':6,'head_dim':128,'sequence_lengths':[32,1024],
            'dtypes':['float16','float32'],'qk_relation':['independent','identical_rms_normalized'],
            'causal':True,'dropout':0},'assigned_chips':0,'optimizer_updates':0,'research_examples':0,
        'test_accessed':False,'scope':'Software numerical arrays only. Installed Math2 forward/backward formulas run on CPU against PyTorch MATH. Explicit FP32 attention call is a prospective workaround. No native compiled behavior,model quality,training change or upstream submission is established.'}
    write_state(out/'plan.json',plan);write_state(public/'plan.json',plan)
    result={'status':'STOPPED_CPU_MATH2_SDPA_REPRO'};records=[]
    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if (time.time()-started>plan['seconds_limit'] or available<544*1024**3
                or min(shutil.disk_usage(out).free,shutil.disk_usage('/').free)<35*1024**3):raise RuntimeError('CPU SDPA fixture reserve/cap')
    try:
        guard();torch.set_num_threads(4);torch.set_num_interop_threads(1)
        spec=importlib.util.spec_from_file_location('isolated_sdk_sdpa_math',a.sdk_source)
        sdk=importlib.util.module_from_spec(spec);spec.loader.exec_module(sdk)
        cases=fixture_cases();gold={}
        for case in cases:
            guard();reference=pytorch_reference(case);gold[case['identity']]=reference
            q,k,v=(case[n] for n in ['q','k','v']);gradient=case['grad_out']
            for policy in ['installed_formula','explicit_fp32_attention']:
                x,y,z=(q,k,v) if policy=='installed_formula' else (q.float(),k.float(),v.float())
                out_value,lse=sdk.compute_attention(x,y,z,None,None,case['scale'],True,0.)
                dq,dk,dv,_=sdk.compute_attention_backward(gradient.to(x.dtype),x,y,z,None,None,
                    out_value,lse,case['scale'],True,0.)
                actual={'output':out_value.to(q.dtype),'q':dq.to(q.dtype),'k':dk.to(k.dtype),'v':dv.to(v.dtype)}
                records.append({'fixture':case['identity'],'policy':policy,'comparison':comparison(reference,actual)})
        fixture_file=out/'fixtures.pt';gold_file=out/'cpu-gold.pt'
        atomic_checkpoint(fixture_file,{'cases':cases});atomic_checkpoint(gold_file,{'records':gold})
        manifest={'identity':plan['identity'],'plan_sha256':file_sha256(out/'plan.json'),
            'fixtures_file':fixture_file.name,'fixtures_sha256':file_sha256(fixture_file),
            'gold_file':gold_file.name,'gold_sha256':file_sha256(gold_file),
            'sdk_python_source_sha256':plan['sdk_python_source_sha256'],
            'fixture_digests':{c['identity']:digest({n:c[n] for n in ['q','k','v','grad_out']}) for c in cases},
            'fixture_metadata':[{k:v for k,v in c.items() if k not in ['q','k','v','grad_out']} for c in cases],
            'fixture_shapes':{c['identity']:list(c['q'].shape) for c in cases},
            'fixture_dtype':{c['identity']:str(c['q'].dtype) for c in cases},'source_sha256':plan['source_sha256']}
        guard()
        if file_sha256(a.sdk_source)!=plan['sdk_python_source_sha256']:raise AssertionError('Installed SDK source changed')
        write_state(out/'manifest.json',manifest);write_state(public/'manifest.json',manifest)
        result={'status':'COMPLETE_CPU_MATH2_SDPA_REPRO','records':records,'fixture_cases':len(cases),
            'manifest_sha256':file_sha256(out/'manifest.json'),'native_execution_qualified':False}
    except BaseException as error:result['error']=str(error);raise
    finally:
        result.update(wall_seconds=time.time()-started,assigned_chips=0,optimizer_updates=0,research_examples=0,
            test_accessed=False,scope=plan['scope']);write_state(out/'report.json',result);write_state(public/'report.json',result)
        print(json.dumps({'status':result['status'],'wall_seconds':result['wall_seconds']}),flush=True)


if __name__=='__main__':main()
