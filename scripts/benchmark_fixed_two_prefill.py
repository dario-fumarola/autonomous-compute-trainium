"""Fixed-curve cooled endpoints: zero-update head-removed prefill timing.

Derived from the qualified benchmark_future_span_prefill driver. Only the
scientific endpoint check changes: use its exact registered cooled update.
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'
if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]

import numpy as np
import torch
import torch.distributed as dist

from compute_trainium.future_span_evaluation import prediction_state
from compute_trainium.future_span_prefill import readout,check_readout
from compute_trainium.future_span_scientific_state import signature
from compute_trainium.parcae_candidate import ParcaeCandidateConfig,matmul_ledger
from compute_trainium.parcae_primitive_sdpa_candidate import ParcaePrimitiveSDPACandidate
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.qualify_native import digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--case',required=True);parser.add_argument('--round',type=int,required=True)
    parser.add_argument('--comm-port',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--corpus');args=parser.parse_args();started=time.time()
    if file_sha256(args.plan)!=args.plan_sha:raise AssertionError('Frozen prefill plan changed')
    plan=json.loads(args.plan.read_text());phase=plan['phases'][args.output.name]
    if (plan['purpose']!='read_only_trained_prefill_cost' or plan['optimizer_updates_allowed']!=0
            or plan['context']!=1024 or plan['warm_forwards']!=100 or plan['timed_forwards']!=100
            or phase!={'case_id':args.case,'round':args.round} or plan['root_comm_ports'][args.output.name]!=args.comm_port):
        raise ValueError('Exact registered prefill phase/port required')
    code=Path(__file__).resolve().parents[1]
    if any(file_sha256(code/f)!=sha for f,sha in plan['worker_source_sha256'].items()):
        raise AssertionError('Exact saved inference source changed')
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(args.comm_port)
    import torch_neuronx
    torch.set_num_threads(2);torch.neuron.set_device(0);dist.init_process_group('neuron')
    rank,world=dist.get_rank(),dist.get_world_size()
    if world!=32:raise ValueError('Exactly32 independently serving cores required')
    args.output.mkdir(exist_ok=True)
    prep=args.root/plan['preparation'];prep_report=prep/'report.json'
    if file_sha256(prep_report)!=plan['preparation_report_sha256']:raise AssertionError('Independent references changed')
    report=json.loads(prep_report.read_text());case=plan['cases'][args.case]
    original_file=args.root/plan['source_main_plan'];original=json.loads(original_file.read_text())
    if file_sha256(original_file)!=plan['source_main_plan_sha256']:raise AssertionError('Original scientific history changed')
    checkpoint=args.root/case['checkpoint']
    if rank==0:
        if file_sha256(checkpoint)!=case['checkpoint_sha256']:raise AssertionError('Original weights changed')
        if any(file_sha256(prep/f)!=sha for f,sha in report['files_sha256'].items()):
            raise AssertionError('CPU input/noise/reference files changed')
    dist.barrier();saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if (saved['signature']!=signature(plan['source_main_plan_sha256'],original,case['source_case'])
            or saved['case_id']!=args.case or saved['update']!=case['source_case']['total_updates']
            or saved['qualification_only'] or saved['stable_trunk']):
        raise ValueError('Exact trained scientific endpoint required')
    if digest({'model':dict(sorted(saved['model'].items())),'optimizer':saved['optimizer']})!=case['full_state_digest']:
        raise AssertionError('Original complete state readback differs')
    state=prediction_state(saved['model']);before=digest(dict(sorted(state.items())))
    config=ParcaeCandidateConfig(**plan['model'])
    model=ParcaePrimitiveSDPACandidate(config,initialize=False,fp16_body=True).eval().requires_grad_(False)
    model.load_state_dict(state,strict=True);del saved,state
    if digest(dict(sorted(model.state_dict().items())))!=before:raise AssertionError('Prediction weights changed on head removal')
    arrays={name:np.load(prep/(name+'.npy'),mmap_mode='r') for name in ['inputs','targets','real-mask','noise']}
    source_indices=[rank,rank+32]
    values=[tuple(torch.from_numpy(np.array(arrays[name][i:i+1],copy=True)).to('neuron')
                  for name in ['inputs','targets','real-mask','noise']) for i in source_indices]
    counts=[int(np.asarray(arrays['real-mask'][i]).sum()) for i in source_indices]
    if min(counts)<1:raise ValueError('No source windows may be discarded or empty')
    model.to('neuron');model.compile_stages('neuron')
    if sum(p.numel() for p in model.parameters())!=config.parameter_count(True):
        raise AssertionError('Deployed parameter count/ties differ')
    scoring=torch.compile(readout,backend='neuron',fullgraph=True,dynamic=False)
    records=[]
    with torch.inference_mode():
        for depth in case['depths']:
            reference=next(r for r in report['records'] if r['case_id']==args.case and r['depth']==depth)
            if reference['prediction_state_digest']!=before:raise AssertionError('CPU reference uses different weights')
            expected_logits=np.load(prep/reference['logits_file'],mmap_mode='r')
            expected_sums=np.load(prep/reference['nll_file'],mmap_mode='r')
            def forward(i):
                tokens,_,_,noise=values[i]
                return model(tokens,noise,total_steps=depth,backward_cap=depth)
            def check(i):
                logits=forward(i);loss,samples=scoring(logits,values[i][1],values[i][2])
                row=check_readout(loss.cpu().item(),samples.cpu(),expected_sums[source_indices[i]],
                    torch.from_numpy(np.array(expected_logits[source_indices[i]:source_indices[i]+1],copy=True)),counts[i])
                del logits;return row
            cold=time.perf_counter();quality=[check(0),check(1)];cold=time.perf_counter()-cold
            dist.barrier();warm=time.perf_counter()
            for j in range(100):
                logits=forward(j%2);torch.neuron.synchronize();del logits
            warm=time.perf_counter()-warm;dist.barrier();times=[]
            for j in range(100):
                begin=time.perf_counter();logits=forward(j%2);torch.neuron.synchronize()
                times.append(time.perf_counter()-begin);del logits
            quality.extend([check(0),check(1)])
            records.append(dict(depth=depth,cold_two_checks_seconds=cold,warm100_seconds=warm,
                timed100_seconds=sum(times),timed_forward_seconds=times,numerical_checks=quality,
                timed_real_input_presentations=50*sum(counts),unique_real_input_positions=sum(counts),
                actual_model_forwards=204,forward_matrix_ops_per_call=matmul_ledger(config,1024,total_steps=depth)['forward_matmul_flops']))
    after=digest(dict(sorted(model.state_dict().items())))
    if after!=before or any(p.grad is not None for p in model.parameters()):
        raise AssertionError('Read-only inference changed weights or created gradients')
    fallback=set(torch_neuronx.get_fallback_ops())
    if fallback-{'aten::_to_copy'}:raise AssertionError('Unexpected prefill fallback: '+str(fallback))
    write_state(args.output/f'rank{rank}.json',dict(rank=rank,prediction_state_digest=after,
        records=records,fallback_ops=sorted(fallback),optimizer_updates=0))
    dist.barrier()
    if rank==0:
        ranks=[json.loads((args.output/f'rank{i}.json').read_text()) for i in range(32)]
        if {r['prediction_state_digest'] for r in ranks}!={before}:raise AssertionError('Read-only serving replicas differ')
        if file_sha256(checkpoint)!=case['checkpoint_sha256']:raise AssertionError('Benchmark modified original file')
        rows=[]
        for depth in case['depths']:
            per_rank=[next(v for v in r['records'] if v['depth']==depth) for r in ranks]
            timings=sorted(t for r in per_rank for t in r['timed_forward_seconds'])
            rows.append(dict(depth=depth,warm_forwards_per_rank=100,timed_forwards_per_rank=100,
                actual_forwards=204*32,timed_forward_median_seconds=statistics.median(timings),
                timed_forward_p95_seconds=timings[int(.95*len(timings))],
                minimum_rank_mean_seconds=min(r['timed100_seconds']/100 for r in per_rank),
                maximum_rank_mean_seconds=max(r['timed100_seconds']/100 for r in per_rank),
                max_rank_timed100_seconds=max(r['timed100_seconds'] for r in per_rank),
                real_input_presentations_per_second=sum(r['timed_real_input_presentations'] for r in per_rank)/max(r['timed100_seconds'] for r in per_rank),
                maximum_cold_two_checks_seconds=max(r['cold_two_checks_seconds'] for r in per_rank),
                maximum_absolute_nll_error=max(abs(q['nll_difference']) for r in per_rank for q in r['numerical_checks']),
                maximum_logit_relative_l2=max(q['logit_relative_l2'] for r in per_rank for q in r['numerical_checks']),
                forward_matrix_ops_per_call=per_rank[0]['forward_matrix_ops_per_call']))
        result=dict(status='PASS_READ_ONLY_NATIVE_PREFILL_TIMING',case_id=args.case,round=args.round,
            plan_sha256=args.plan_sha,checkpoint_sha256=case['checkpoint_sha256'],prediction_state_digest=before,
            deployed_parameters=config.parameter_count(True),head_removed=True,weights_unchanged=True,
            unique_windows=64,context=1024,world=32,records=rows,optimizer_updates=0,
            gradients_created=False,test_accessed=False,wall_seconds=time.time()-started,
            nominal_forward_matrix_ops=32*204*sum(r['forward_matrix_ops_per_call'] for r in rows),
            physical_chips=sorted({int(c)//4 for c in os.environ['RESEARCH_VISIBLE_CORES'].split(',')}),
            scope=plan['scope'])
        write_state(args.output/'report.json',result)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
