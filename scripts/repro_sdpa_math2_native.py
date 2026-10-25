"""Compiled SDPA fixture capture on32 owned ranks;zero optimizer updates."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'
os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]

import torch
import torch.distributed as dist
from torch.nn import functional as F

from compute_trainium.parcae_math_sdpa_candidate import math_sdpa
from compute_trainium.training import atomic_checkpoint,file_sha256
from scripts.qualify_native import digest


def native_sdpa(q,k,v):return F.scaled_dot_product_attention(q,k,v,is_causal=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True)
    p.add_argument('--comm-port',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus')
    a=p.parse_args();started=time.time()
    if file_sha256(a.plan)!=a.plan_sha:raise AssertionError('SDPA fixture plan changed')
    plan=json.loads(a.plan.read_text())
    if plan['root_comm_ports'].get(a.output.name)!=a.comm_port:raise ValueError('Owned registered operator phase required')
    code_root=Path(__file__).resolve().parents[1]
    if any(file_sha256(code_root/f)!=sha for f,sha in plan['worker_source_sha256'].items()):raise AssertionError('Frozen operator source changed')
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
    import torch_neuronx
    import torch_neuronx._sdpa_math as sdk_math
    if file_sha256(sdk_math.__file__)!=plan['fixture_manifest']['sdk_python_source_sha256']:
        raise AssertionError('Native and inspected SDK source differ')
    torch.set_num_threads(2);torch.neuron.set_device(0);dist.init_process_group('neuron')
    rank,world=dist.get_rank(),dist.get_world_size()
    if world!=32:raise ValueError('Explicit32-rank numerical fixture replication required')
    a.output.mkdir(exist_ok=True);info=plan['fixture_manifest'];directory=a.root/plan['fixture_directory']
    fixture_file=directory/info['fixtures_file']
    if file_sha256(fixture_file)!=info['fixtures_sha256']:raise AssertionError('Exact numerical inputs changed')
    cases=torch.load(fixture_file,map_location='cpu',mmap=True,weights_only=False)['cases']
    functions={'installed_operator':torch.compile(native_sdpa,backend='neuron',fullgraph=True,dynamic=False),
        'explicit_fp32_attention':torch.compile(math_sdpa,backend='neuron',fullgraph=True,dynamic=False)}
    rows=[]
    for case in cases:
        if digest({n:case[n] for n in ['q','k','v','grad_out']})!=info['fixture_digests'][case['identity']]:
            raise AssertionError('Numerical input digest differs')
        for policy,function in functions.items():
            begin=time.time();q,k,v=[case[n].to('neuron').detach().requires_grad_() for n in ['q','k','v']]
            out=function(q,k,v);out.backward(case['grad_out'].to('neuron'))
            packet={'output':out.detach().cpu(),'q':q.grad.detach().cpu(),'k':k.grad.detach().cpu(),'v':v.grad.detach().cpu()}
            if any(not torch.isfinite(t).all() for t in packet.values()):raise FloatingPointError('Nonfinite SDPA fixture')
            key=case['identity']+'-'+policy;row={'fixture':case['identity'],'policy':policy,'digest':digest(packet),
                'wall_seconds':time.time()-begin}
            path=a.output/(key+f'-rank{rank}.pt');atomic_checkpoint(path,packet)
            row.update(rank=rank,file=path.name,sha256=file_sha256(path))
            rows.append(row);dist.barrier()
    (a.output/f'replica-rank{rank}.json').write_text(json.dumps({'digests':[(r['fixture'],r['policy'],r['digest']) for r in rows],
        'records':rows,'fallback_ops':list(torch_neuronx.get_fallback_ops())})+'\n');dist.barrier()
    if rank==0:
        replicas=[json.loads((a.output/f'replica-rank{i}.json').read_text()) for i in range(32)]
        identical=all(r['digests']==replicas[0]['digests'] for r in replicas)
        if any(set(r['fallback_ops'])-{'aten::_to_copy'} for r in replicas):raise AssertionError('Unexpected operator fallback')
        report={'status':'CAPTURED_COMPILED_SDPA_NUMERICAL_FIXTURES','plan_sha256':a.plan_sha,
            'records':[r for replica in replicas for r in replica['records']],
            'all32_replica_packets_equal':identical,'optimizer_updates':0,'research_examples':0,
            'test_accessed':False,'wall_seconds':time.time()-started,
            'scope':'Actual compiled native SDPA outputs and input gradients on numerical operator fixtures. Exact same CPU-prepared inputs replicated on32 owned logical cores. Independent CPU comparison required;no corpus or optimizer updates.'}
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
