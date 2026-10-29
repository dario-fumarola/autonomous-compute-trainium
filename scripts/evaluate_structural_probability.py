"""Frozen supported-exit structural readout; no fitting or adaptive selection."""
import argparse,json,os,time
from pathlib import Path
if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]

def truncated_arrays(xs,ys,ms,meta):
    import numpy as np
    x=np.full_like(xs,2);y=np.full_like(ys,2);m=np.zeros_like(ms)
    for i,row in enumerate(meta):
        length=row['prefix_tokens'];position=row['target_start']
        if length<32 or position!=length-1 or ms[i].sum()!=1 or not ms[i,position]:raise AssertionError('Exact prefix/target alignment')
        if row['kind']=='structural_long' and row['opener_distance']<64:raise AssertionError('Long opener must be removed')
        if row['kind']=='structural_short' and row['opener_distance']>=16:raise AssertionError('Short opener must remain')
        x[i,:32]=xs[i,length-32:length];y[i,31]=ys[i,position];m[i,31]=True
    return x,y,m

def main(a):
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch_neuronx
    from torch import nn
    from compute_trainium.probability_model_factory import probability_model
    from compute_trainium.probability_intervention import ProposalReadout
    from compute_trainium.stable_target_loss import stable_target_log_probs
    from compute_trainium.training import file_sha256
    p=json.loads(a.plan.read_text());src=Path(__file__).resolve().parents[1]
    if file_sha256(a.plan)!=a.plan_sha or any(file_sha256(src/f)!=h for f,h in p['worker_source_sha256'].items()):raise AssertionError('Frozen implementation changed')
    panel=a.root/p['panel_relative'];receipt=json.loads((panel/'report.json').read_text())
    if file_sha256(panel/'report.json')!=p['panel_report_sha256'] or any(file_sha256(panel/f)!=h for f,h in receipt['files'].items()):raise AssertionError('Frozen source panel changed')
    meta=json.loads((panel/'PRIVATE-metadata.json').read_text());xs=np.load(panel/'inputs.npy');ys=np.load(panel/'targets.npy');ms=np.load(panel/'mask.npy')
    panels={'full':(xs,ys,ms),'last32':truncated_arrays(xs,ys,ms,meta)}
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);torch.set_num_threads(1)
    if world!=64:raise AssertionError('Registered64rank topology')
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port);torch.neuron.set_device(0);dist.init_process_group('neuron')
    a.output.mkdir(parents=True,exist_ok=True);started=time.time();torch.manual_seed(p['seed']);model=probability_model(p,explicit_attention=True).to('neuron')
    class Scorer(nn.Module):
        def __init__(self,depth):super().__init__();self.reader=ProposalReadout(model,depth,'expected')
        def forward(self,x,y):return -stable_target_log_probs(self.reader(x),y)
    scorers={k:torch.compile(Scorer(k).to('neuron'),backend='neuron',fullgraph=True,dynamic=False) for k in (0,1)}
    rows=[];checks=[];batch=p['batch']
    for point in p['structural_points']:
        checkpoint=a.root/'runs'/point['phase']/('PRIVATE-u'+str(point['update'])+'.pt')
        if rank==0 and file_sha256(checkpoint)!=point['checkpoint_sha256']:raise AssertionError('Audited model bytes changed')
        dist.barrier();saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);weights=saved['model'];model.load_state_dict(weights)
        for condition,(xx,yy,mm) in panels.items():
            for left in range(0,len(meta),world*batch):
                indices=[left+rank*batch+j for j in range(batch)];safe=[i if i<len(meta) else 0 for i in indices]
                x=torch.from_numpy(np.asarray(xx[safe],dtype=np.int64));y=torch.from_numpy(np.asarray(yy[safe],dtype=np.int64));mask=torch.from_numpy(np.asarray(mm[safe],dtype=np.bool_))
                with torch.no_grad():losses=scorers[point['depth']](x.to('neuron'),y.to('neuron')).cpu()
                if left==0 and rank in (0,32):
                    torch.set_num_threads(8);cpu=probability_model(p).double();cpu.load_state_dict(weights)
                    with torch.no_grad():
                        logits=cpu(x,extra_steps=point['depth']);expected=torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').view_as(y)
                    error=float((losses[mask]-expected[mask]).abs().max());checks.append(dict(label=point['label'],condition=condition,first_index=safe[0],maximum_target_loss_error=error,reference_precision='FP64'))
                    if error>p['evaluation_maximum_loss_error']:raise AssertionError('Independent FP64 target-loss gate failed')
                    del cpu,logits,expected;torch.set_num_threads(1)
                for j,i in enumerate(indices):
                    if i>=len(meta):continue
                    selected=losses[j][mask[j]];item=meta[i]
                    if len(selected)!=1:raise AssertionError('Single authentic target required')
                    rows.append(dict(label=point['label'],condition=condition,id=item['id'],group=item['group'],kind=item['kind'],targets=1,loss_sum=float(selected[0])))
                if rank==0:(a.output/'progress.json').write_text(json.dumps(dict(label=point['label'],condition=condition,left=left,seconds=time.time()-started)))
        del saved,weights
    (a.output/f'PRIVATE-rank{rank}-scores.json').write_text(json.dumps(rows));(a.output/f'cpu-checks-rank{rank}.json').write_text(json.dumps(checks));(a.output/f'fallback-rank{rank}.json').write_text(json.dumps(list(torch_neuronx.get_fallback_ops())));dist.barrier()
    if rank==0:
        if any(set(json.loads((a.output/f'fallback-rank{r}.json').read_text()))-{'aten::_to_copy'} for r in range(world)):raise AssertionError('Unexpected fallback')
        allchecks=[c for r in range(world) for c in json.loads((a.output/f'cpu-checks-rank{r}.json').read_text())]
        result=dict(status='COMPLETE_STRUCTURAL_PROBABILITY_READOUT',world=world,wall_seconds=time.time()-started,models=len(p['structural_points']),conditions=list(panels),cases=len(meta),source_groups=receipt['paired_source_groups'],independent_cpu_checks=allchecks,maximum_cpu_target_loss_error=max(c['maximum_target_loss_error'] for c in allchecks),optimizer_updates=0,protected_groups_scored=0)
        (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--root',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--corpus');q.add_argument('--plan',type=Path,required=True);q.add_argument('--plan-sha',required=True);q.add_argument('--comm-port',type=int,required=True);main(q.parse_args())
