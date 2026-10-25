"""Actual-size CPU integration rehearsal for the new loss and audit adapters."""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA','0')
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel

from compute_trainium.future_span import FutureSpanCandidate,FutureSpanConfig
from compute_trainium.future_span_training import FutureSpanObjective,objective_coefficients,training_window
from compute_trainium.parcae_candidate import ParcaeCandidateConfig,matmul_ledger
from compute_trainium.queue import write_state
from compute_trainium.training import PackedCorpus,file_sha256
from scripts.audit_future_span_qualification import construct_reference,gradient_check
from scripts.qualify_native import digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();started=time.time()
    preparation=args.root/'future-span-qualification-preparation-v1'
    manifest=json.loads((preparation/'manifest.json').read_text())
    qplan=args.root/'future-span-qualification-v1/plan.json';plan=json.loads(qplan.read_text())
    out=args.root/'future-span-cpu-adapter-rehearsal-v1';public=Path('results/future_span_qualification_20261025/cpu-adapter-rehearsal')
    out.mkdir(exist_ok=False);public.mkdir(parents=True,exist_ok=False)
    registration={'registered_unix':started,'qualification_plan_sha256':file_sha256(qplan),
        'source_sha256':{**plan['source_sha256'],'scripts/rehearse_future_span_cpu.py':file_sha256(__file__)},
        'source_window':0,'depths':[1,2],'objective_weight':.1,'seconds_limit':300,
        'assigned_chips':0,'optimizer_updates':0,'test_accessed':False,
        'scope':'Actual-size FP32 CPU adapter integration,not native qualification. Same immutable qualification '
        'initializer,one real TRAIN window,causal model and independent pinned upstream body/dense BCE. '
        'No scientific updates or held-out scoring.'}
    write_state(out/'registration.json',registration);write_state(public/'registration.json',registration)
    records=[];result={'status':'STOPPED_FUTURE_SPAN_CPU_ADAPTER_REHEARSAL'}
    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if (available<544*1024**3 or min(shutil.disk_usage(out).free,shutil.disk_usage('/').free)<35*1024**3
                or time.time()-started>registration['seconds_limit']):raise RuntimeError('CPU adapter rehearsal cap/reserve')
    try:
        guard();torch.set_num_threads(8)
        sys.path.insert(0,str(args.root/'upstream-parcae-69284c1'))
        config=ParcaeCandidateConfig(**manifest['model']);span=FutureSpanConfig(**manifest['future_config'])
        initial_file=preparation/manifest['initial_file']
        if file_sha256(initial_file)!=manifest['initial_sha256']:raise AssertionError('Prepared initializer changed')
        initial=torch.load(initial_file,map_location='cpu',mmap=True,weights_only=False)
        packed=PackedCorpus.from_mapped(args.root/manifest['data']['packed'],1024,23,manifest['data']['corpus_sha256'])
        candidate=FutureSpanCandidate(config,span,initialize=False,fp16_body=False);candidate.load_state_dict(initial['model'])
        reference=construct_reference(config,initial['model'])
        if digest(dict(sorted(reference.state_dict().items())))!=manifest['model_digest']:raise AssertionError('Independent audit adapter altered the initializer')
        objective=FutureSpanObjective(candidate)
        values=training_window(packed,0,config,span,manifest['state_seed']);total=int(values[2].sum());anchors=int(values[-1].sum())
        coefficients=objective_coefficients(total,anchors,.1,1)
        matrix_work=0
        for depth in registration['depths']:
            guard();candidate.zero_grad();reference.zero_grad();hidden={}
            with sdpa_kernel(SDPBackend.MATH):
                value,parts=objective(*values,coefficients,depth,depth);value.backward()
                reference.initialize_state=lambda _:values[3].clone()
                hook=reference.transformer.ln_f.register_forward_hook(lambda m,ins,out:hidden.update(value=out))
                logits=reference(values[0],return_logits=True,num_steps_pair=torch.tensor([0,depth]))['logits']
                aux=reference.future_lm_head(hidden['value'][:,reference.future_positions])
                truth=torch.zeros_like(aux);valid=[]
                for j,pos in enumerate(span.positions):
                    ids=set(map(int,packed.tokens[pos+2:pos+span.horizon+1]));truth[0,j,list(ids)]=1
                    valid.append(bool((packed.domains[pos:pos+span.horizon+1]>0).all()))
                ntp=(F.cross_entropy(logits.flatten(0,1),values[1].flatten(),reduction='none')*values[2].flatten()).sum()
                future=(F.binary_cross_entropy_with_logits(aux,truth,reduction='none').sum(-1)*torch.tensor(valid)[None]).sum()/(span.horizon-1)
                expected=ntp/total+.1*future/anchors;expected.backward();hook.remove()
            comparison=gradient_check({n:p.grad for n,p in reference.named_parameters()},
                                      {n:p.grad for n,p in candidate.named_parameters()})
            difference=float(value.detach()-expected.detach())
            if not comparison['passes_two_percent_gate'] or abs(difference)>.003:raise RuntimeError('CPU adapter integration mismatch')
            records.append({'depth':depth,'gradient':comparison,'combined_loss_difference':difference})
            matrix_work+=2*(matmul_ledger(config,1024,total_steps=depth,backward_cap=depth)['training_matmul_flops']
                +6*len(span.positions)*config.dim*config.vocab_size)
        if any(file_sha256(p)!=sha for p,sha in registration['source_sha256'].items()):raise AssertionError('Rehearsal source changed')
        result={'status':'PASS_ACTUAL_SIZE_CPU_FUTURE_ADAPTER_REHEARSAL','records':records,
            'diagnostic_matrix_operations':matrix_work,'unique_real_targets':total,
            'diagnostic_real_target_presentations':4*total,'native_qualification_established':False}
    except BaseException as error:result['error']=str(error);raise
    finally:
        result.update(assigned_chips=0,optimizer_updates=0,test_accessed=False,wall_seconds=time.time()-started,scope=registration['scope'])
        write_state(out/'report.json',result);write_state(public/'report.json',result)
        print(json.dumps({'status':result['status'],'wall_seconds':result['wall_seconds']}),flush=True)


if __name__=='__main__':main()
