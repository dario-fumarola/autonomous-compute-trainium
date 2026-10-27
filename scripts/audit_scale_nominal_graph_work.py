"""Independent ATen dispatch matrix counts on actual-shape meta software arrays.

No source content, numeric model values, research examples, device kernels or
learning updates. Autocast is disabled for unsupported meta devices; this is
only a nominal shape/FMA count, never a precision or hardware-execution audit.
"""
import argparse,contextlib,json,time
from pathlib import Path


def audit():
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from compute_trainium.scale_pilot_size import profiled_inventory
    from compute_trainium.parcae_candidate import ParcaeCandidate,ParcaeCandidateConfig,matmul_ledger
    from compute_trainium.parcae_primitive_sdpa_candidate import ParcaePrimitiveSDPACandidate
    from compute_trainium.parcae_optimizer_candidate import parameter_groups
    from compute_trainium.parcae_optimizer_precise import muon_tensor_step_fp32
    from compute_trainium.parcae_work_accounting import muon_update_matmuls
    from compute_trainium.scale_work_ledger import fixed_update_work,work_matched_updates
    from compute_trainium.training import file_sha256
    started=time.time();records=[];original_autocast=torch.autocast
    try:
        torch.autocast=lambda *args,**kwargs:contextlib.nullcontext()
        for row in profiled_inventory('medium'):
            config=ParcaeCandidateConfig(**row['config']);recurrent=row['role']=='fixed_two';depth=2 if recurrent else 1
            with torch.device('meta'):
                model=ParcaePrimitiveSDPACandidate(config,recurrent=recurrent,initialize=False,fp16_body=True)
                tokens=torch.empty((1,1024),dtype=torch.long);state=torch.empty((1,1024,config.dim)) if recurrent else None
            with FlopCounterMode(display=False) as counter:
                logits=model(tokens,state,total_steps=depth,backward_cap=depth)
                forward=counter.get_total_flops();logits.sum().backward();training=counter.get_total_flops()
            observed_operations={str(k):v for k,v in counter.get_flop_counts()['Global'].items()}
            declared=matmul_ledger(config,1024,1,recurrent=recurrent,total_steps=depth,backward_cap=depth)
            if forward!=declared['forward_matmul_flops'] or training!=declared['training_matmul_flops']:
                raise AssertionError('Independent complete forward/backward matrix counts differ from ledger')
            with torch.device('meta'),torch.no_grad():
                optimizer_model=ParcaeCandidate(config,recurrent=recurrent,initialize=False)
                groups=parameter_groups(optimizer_model.named_parameters(),gate_policy='release_muon')
                with FlopCounterMode(display=False) as counter:
                    for group in groups:
                        if group['kind']!='muon':continue
                        packed=torch.stack(group['params']);shape=packed.shape;reduction=-1 if shape[-2]>=shape[-1] else -2
                        vshape=(shape[0],shape[-2],1) if reduction==-1 else (shape[0],1,shape[-1])
                        muon_tensor_step_fp32(torch.empty_like(packed),packed,torch.empty_like(packed),torch.empty(vshape),
                            torch.tensor(.95),torch.tensor(.008),torch.tensor(.2),torch.tensor(.95),group['ns_steps'],reduction)
                observed_optimizer=counter.get_total_flops()
            declared_optimizer=muon_update_matmuls(optimizer_model.named_parameters(),gate_policy='release_muon')
            ledger=fixed_update_work(config,row['role'])
            if observed_optimizer!=declared_optimizer or training*512+observed_optimizer!=ledger['total_per_update']:
                raise AssertionError('Owner-only optimizer or complete global update matrix ledger differs')
            records.append(dict(role=row['role'],parameters=row['parameters'],context=1024,global_windows=512,
                forward_per_window=forward,training_per_window=training,neural_operations=observed_operations,
                owner_optimizer_per_update=observed_optimizer,total_per_update=ledger['total_per_update'],
                owners_per_parameter_group=1,owner_parameter_groups=sum(g['kind']=='muon' for g in groups),
                parameter_group_owners_not_replicated_across64_workers=True))
    finally:torch.autocast=original_autocast
    r,o=records;match=work_matched_updates(r['total_per_update'],9554,o['total_per_update'])
    if match['updates']!=12939 or match['candidate_total']<=2**63:raise AssertionError('Frozen endpoints/Python integer scope changed')
    files=['scripts/audit_scale_nominal_graph_work.py','compute_trainium/scale_work_ledger.py',
           'compute_trainium/parcae_candidate.py','compute_trainium/parcae_primitive_sdpa_candidate.py',
           'compute_trainium/parcae_optimizer_precise.py','compute_trainium/scale_model_sharding.py']
    return dict(status='PASS_INDEPENDENT_META_MATRIX_GRAPH_COUNTS',records=records,frozen_main_match=match,
        source_sha256={f:file_sha256(f) for f in files},torch_version=torch.__version__,wall_seconds=time.time()-started,
        assigned_chips=0,research_examples=0,optimizer_updates=0,
        scope='ATen mm/bmm dispatch counts actual full forward/backward and complete FP32 polar matrix algorithm shapes. Same-shape kernels across512 windows,once per owner optimizer group. No assertion about Native compiler padding,recomputation,collectives,elementwise/AdamW work,precision throughput or chip-time matching; these remain separate.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve every audit')
    result=audit();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','source_sha256']}),flush=True)
