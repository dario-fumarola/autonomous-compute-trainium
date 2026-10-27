"""Zero-update Native software fixture; no research examples or predictor."""
import argparse
import json
import os
from pathlib import Path
import time

if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--comm-port',type=int,required=True)
    p.add_argument('--corpus');a=p.parse_args();started=time.time()
    import torch
    import torch.distributed as dist
    from compute_trainium.training import file_sha256
    from compute_trainium.rejected_replica_evidence import enforce_replica_agreement,compare_inventories
    from compute_trainium.future_span_scientific_state import SCIENCE,validate_checkpoint
    from compute_trainium.head_free_fixed_two_v2_gate import FIXTURE_STATUS,fixture_gate
    from scripts.qualify_native import digest
    if file_sha256(a.plan)!=a.plan_sha:raise AssertionError('Frozen numerical fixture plan changed')
    plan=json.loads(a.plan.read_text())
    if (plan['purpose']!='baseline_native_qualification_only' or not plan['rejected_state_evidence_required']
            or plan['fixture_phase']!=a.output.name or plan['root_comm_ports'].get(a.output.name)!=a.comm_port):
        raise ValueError('Exact numerical-only fixture phase/port required')
    code_root=Path(__file__).resolve().parents[1]
    if any(file_sha256(code_root/f)!=sha for f,sha in plan['worker_source_sha256'].items()):
        raise AssertionError('Frozen fixture worker source changed')
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
    import torch_neuronx
    torch.set_num_threads(2);torch.neuron.set_device(0);dist.init_process_group('neuron')
    rank,world=dist.get_rank(),dist.get_world_size()
    if world!=32:raise ValueError('Original32-rank numerical geometry required')
    a.output.mkdir(exist_ok=True)
    def state(different=False):
        # Explicit software arrays,never source text,research targets or a model.
        source={'model':{'w':torch.tensor([.25,.5],dtype=torch.float32)},
                'optimizer':{'m':torch.tensor([.125+(1 if different else 0),.25],dtype=torch.float32),'step':0}}
        result={'model':{'w':source['model']['w'].to('neuron')},
                'optimizer':{'m':source['optimizer']['m'].to('neuron'),'step':0}}
        if result['model']['w'].device.type!='neuron' or result['optimizer']['m'].device.type!='neuron':
            raise AssertionError('Actual Native fixture tensors required')
        if digest(result)!=digest(source):raise AssertionError('Native fixture differs from exact CPU software arrays')
        return result
    good=state();good_digest=digest(good)
    (a.output/f'agree-rank{rank}.json').write_text(json.dumps({'digest':good_digest})+'\n');dist.barrier()
    agreeing=[json.loads((a.output/f'agree-rank{i}.json').read_text())['digest'] for i in range(world)]
    if len(set(agreeing))!=1:raise AssertionError('Numerical fixture cannot establish Native readback agreement')
    def forbidden():raise AssertionError('Agreeing evidence guard unexpectedly touched state or a barrier')
    enforce_replica_agreement(a.output,update=1,rank=rank,digests=agreeing,
        state_factory=forbidden,digest_fn=digest,barrier=forbidden,failure_message='unexpected agreement rejection')
    bad=state(different=rank==1);bad_digest=digest(bad)
    (a.output/f'disagree-rank{rank}.json').write_text(json.dumps({'digest':bad_digest})+'\n');dist.barrier()
    disagreement=[json.loads((a.output/f'disagree-rank{i}.json').read_text())['digest'] for i in range(world)]
    if not (disagreement[1]!=disagreement[0] and all(v==disagreement[0] for i,v in enumerate(disagreement) if i!=1)):
        raise AssertionError('Fixture must differ only in the declared rank1 numerical moment')
    rejected=False
    try:
        enforce_replica_agreement(a.output,update=1,rank=rank,digests=disagreement,
            state_factory=lambda:bad,digest_fn=digest,barrier=dist.barrier,
            failure_message='Deliberate numerical fixture replica rejection')
    except AssertionError as error:
        if not str(error).startswith('Deliberate numerical fixture replica rejection'):raise
        rejected=True
    (a.output/f'verified-rank{rank}.json').write_text(json.dumps({'rejected':rejected})+'\n');dist.barrier()
    if rank==0:
        notes=[json.loads((a.output/f'rejected-u1-rank{i}.json').read_text()) for i in [0,1]]
        if any(n['status']!='CAPTURED_PRIVATE_REJECTED_STATE' or not n['repeated_readback_matches_original_digest'] for n in notes):
            raise AssertionError('Actual private rejected-state capture/readback failed')
        snapshots=list(a.output.glob('rejected-u1-rank*.pt'))
        if len(snapshots)!=2:raise AssertionError('Exactly two bounded private fixture snapshots required')
        for note in notes:
            file=a.output/note['snapshot_file']
            if file_sha256(file)!=note['snapshot_sha256']:raise AssertionError('Private fixture snapshot changed')
            payload=torch.load(file,map_location='cpu',mmap=True,weights_only=False)
            try:
                validate_checkpoint(payload,'fixture',{},dict(purpose=SCIENCE),expected_targets=0,
                    expected_domains=[],expected_matrix_ops=0,expected_anchors=0)
            except ValueError:pass
            else:raise AssertionError('Rejected fixture snapshot was accepted for science')
        change=compare_inventories(notes[0]['inventory'],notes[1]['inventory'])
        row=dict(status=FIXTURE_STATUS,purpose='rejected_replica_numerical_fixture_only',plan_sha256=a.plan_sha,
            world=32,optimizer_updates=0,model_forwards=0,research_examples=0,
            native_tensor_devices_verified=True,native_cpu_fixture_bits_match=True,
            agreement_path_no_state_factory_or_extra_barrier=True,
            deliberate_rejection_verified_all32=all(json.loads((a.output/f'verified-rank{i}.json').read_text())['rejected'] for i in range(world)),
            selected_private_ranks=[0,1],private_snapshot_count=len(snapshots),all_private_snapshot_readbacks_match=True,
            all_private_snapshots_rejected_by_scientific_validator=True,changed_leaf_paths=change['changed_leaf_paths'],
            raw_states_remain_private=True,scientific_fit_admitted=False,terminal_alpha_spent=0,
            wall_seconds=time.time()-started,private_snapshot_note_sha256=[file_sha256(a.output/f'rejected-u1-rank{i}.json') for i in [0,1]],
            scope='Numerical arrays only,zero predictor forwards or optimizer updates. Expected rejection is caught only in this isolated qualification fixture. Scientific workers still fail on any replica mismatch. No cause or repair of the unsaved earlier mismatch is established.')
        fixture_gate(row,a.plan_sha);(a.output/'report.json').write_text(json.dumps(row,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
