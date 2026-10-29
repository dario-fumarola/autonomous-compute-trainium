"""Registered supported-budget/readout interventions after all mature audits."""
import argparse,json,os,time
from pathlib import Path

if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def main(a):
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch_neuronx
    from torch import nn
    from compute_trainium.model import ModelConfig
    from compute_trainium.probability_workspace import ProbabilityWorkspace
    from compute_trainium.probability_intervention import ProposalReadout
    from compute_trainium.stable_target_loss import stable_target_log_probs
    from compute_trainium.training import file_sha256
    p=json.loads(a.plan.read_text());src=Path(__file__).resolve().parents[1]
    if file_sha256(a.plan)!=a.plan_sha or any(file_sha256(src/f)!=h for f,h in p['worker_source_sha256'].items()):raise AssertionError('Frozen evaluation changed')
    for arm in p['arms']:
        audit=a.root/p['identity']/('audit-'+arm)/'report.json'
        if json.loads(audit.read_text())['status']!='PASS_PROBABILITY_SCREEN_MATURE_AUDITS':raise AssertionError('Every scientific arm must pass before quality scoring')
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);torch.set_num_threads(1)
    if world!=64:raise ValueError('Registered topology')
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port);torch.neuron.set_device(0);dist.init_process_group('neuron')
    a.output.mkdir(parents=True,exist_ok=True);start=time.time();c=ModelConfig(**p['model_config']);torch.manual_seed(p['seed'])
    model=ProbabilityWorkspace(c,explicit_attention=True);initial={n:v.detach().clone() for n,v in model.state_dict().items()};model.to('neuron')
    permutation=torch.randperm(c.vocab_size,generator=torch.Generator().manual_seed(p['permutation_seed']))
    class Scorer(nn.Module):
        def __init__(self,reader):super().__init__();self.reader=reader
        def forward(self,x,y):return -stable_target_log_probs(self.reader(x),y)
    variants=sorted({(v['depth'],v['proposal'],v.get('permute',False)) for point in p['evaluation_points'] for v in point['variants']})
    scorers={key:torch.compile(Scorer(ProposalReadout(model,key[0],key[1],permutation if key[2] else None)).to('neuron'),backend='neuron',fullgraph=True,dynamic=False) for key in variants}
    evaluation=a.root/p['evaluation_relative'];broad=np.load(evaluation/'broad-tokens.npy',mmap_mode='r')
    panels={'broad':(broad[:,:-1],broad[:,1:],np.ones((len(broad),512),dtype=np.bool_),json.loads((evaluation/'PRIVATE-broad-metadata.json').read_text())),
            'dependency':(np.load(evaluation/'dependency-inputs.npy',mmap_mode='r'),np.load(evaluation/'dependency-targets.npy',mmap_mode='r'),np.load(evaluation/'dependency-mask.npy',mmap_mode='r'),json.loads((evaluation/'PRIVATE-dependency-metadata.json').read_text()))}
    rows=[];checks=[];batch=p['batch']
    for point in p['evaluation_points']:
        if point['arm']=='initial':weights=initial
        else:
            phase=p['arms'][point['arm']]['phase'];path=a.root/'runs'/phase/f"PRIVATE-u{point['update']}.pt"
            if rank==0:
                receipt=json.loads((a.root/p['identity']/('audit-'+point['arm'])/'report.json').read_text())
                hashes={r['checkpoint_sha256'] for r in receipt['rows'] if r['update']==point['update']}
                if hashes!={file_sha256(path)}:raise AssertionError('Audited checkpoint bytes changed')
            dist.barrier()
            saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True);weights=saved['model']
        model.load_state_dict(weights)
        for variant in point['variants']:
            key=(variant['depth'],variant['proposal'],variant.get('permute',False));reader=scorers[key]
            label=point['label']+'-'+variant['label']
            for panel,(xs,ys,ms,meta) in panels.items():
                for left in range(0,len(xs),world*batch):
                    indices=[left+rank*batch+j for j in range(batch)];valid=[i<len(xs) for i in indices];safe=[i if i<len(xs) else 0 for i in indices]
                    x=torch.from_numpy(np.asarray(xs[safe],dtype=np.int64));y=torch.from_numpy(np.asarray(ys[safe],dtype=np.int64));mask=torch.from_numpy(np.asarray(ms[safe],dtype=np.bool_))
                    with torch.no_grad():losses=reader(x.to('neuron'),y.to('neuron')).cpu()
                    # Independent CPU SDPA/CE checks include all broad domains,
                    # and the first dependency batch for the content intervention.
                    check=(panel=='broad' and ((left==0 and rank in (0,32)) or (left==256 and rank==0))) or (panel=='dependency' and left==0 and rank==0 and key[2])
                    if check:
                        torch.set_num_threads(8);reference=ProbabilityWorkspace(c).double();reference.load_state_dict(weights)
                        cpu=ProposalReadout(reference,key[0],key[1],permutation if key[2] else None)
                        with torch.no_grad():
                            logits=cpu(x);expected=torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').view_as(y)
                        error=float((losses[mask]-expected[mask]).abs().max())
                        checks.append(dict(label=label,panel=panel,first_index=safe[0],maximum_target_loss_error=error,reference_precision='FP64'))
                        (a.output/f'cpu-checks-rank{rank}.json').write_text(json.dumps(checks))
                        if error>p['evaluation_maximum_loss_error']:raise AssertionError(f'Independent CPU readout parity failed: {label}, panel={panel}, index={safe[0]}, error={error}')
                        del reference,cpu,logits,expected;torch.set_num_threads(1)
                    for j,i in enumerate(indices):
                        if not valid[j]:continue
                        item=meta[i];selected=losses[j][mask[j]]
                        rows.append(dict(label=label,panel=panel,index=i,id=item['id'],group=item['group'],kind=item['kind'],
                            domain=item.get('domain'),cohort=item.get('cohort'),loss_sum=float(selected.double().sum()),targets=len(selected),first_loss=float(selected[0])))
                    if rank==0:(a.output/'progress.json').write_text(json.dumps(dict(label=label,panel=panel,left=left,seconds=time.time()-start)))
        if point['arm']!='initial':del saved,weights
    (a.output/f'PRIVATE-rank{rank}-scores.json').write_text(json.dumps(rows));(a.output/f'cpu-checks-rank{rank}.json').write_text(json.dumps(checks))
    (a.output/f'fallback-rank{rank}.json').write_text(json.dumps(list(torch_neuronx.get_fallback_ops())));dist.barrier()
    if rank==0:
        if any(set(json.loads((a.output/f'fallback-rank{r}.json').read_text()))-{'aten::_to_copy'} for r in range(world)):raise AssertionError('Unexpected evaluation fallback')
        allchecks=[x for r in range(world) for x in json.loads((a.output/f'cpu-checks-rank{r}.json').read_text())]
        report=dict(status='COMPLETE_PROBABILITY_SCREEN_EVALUATION',world=world,wall_seconds=time.time()-start,
            checkpoint_points=len(p['evaluation_points']),readout_variants=sum(len(x['variants']) for x in p['evaluation_points']),
            broad_windows=len(broad),dependency_cases=len(panels['dependency'][0]),independent_cpu_checks=allchecks,
            maximum_cpu_target_loss_error=max(x['maximum_target_loss_error'] for x in allchecks),optimizer_updates=0,protected_groups_scored=0)
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus')
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--comm-port',type=int,required=True);main(p.parse_args())
