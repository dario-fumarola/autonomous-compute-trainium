"""Concurrent all-chip operator timing with independent CPU gradients."""
import argparse,json,os,time,statistics
from pathlib import Path
if 'RESEARCH_VISIBLE_CORES' in os.environ:
    os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]


def main(a):
    import torch
    import torch.distributed as dist
    import torch_neuronx
    from torch.nn import functional as F
    from compute_trainium.block_execution_probe import CASES,Fixture
    from compute_trainium.training import file_sha256
    from scripts.qualify_trained_gradient import compare_gradients
    p=json.loads(a.plan.read_text());source=Path(__file__).resolve().parents[1]
    if file_sha256(a.plan)!=a.plan_sha or any(file_sha256(source/f)!=h for f,h in p['worker_source_sha256'].items()):raise AssertionError('Frozen operator probe changed')
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);case=rank%8
    if world!=64 or p['purpose']!='block_execution_software_probe':raise ValueError('Software fixture only')
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
    torch.neuron.set_device(0);dist.init_process_group('neuron');a.output.mkdir(exist_ok=True,parents=True)
    fixture=Fixture(case).to('neuron');native=fixture.input().to('neuron').requires_grad_()
    compiled=torch.compile(lambda x:fixture(x).square().mean(),backend='neuron',fullgraph=True,dynamic=False)
    started=time.time();samples=[];check=None
    for iteration in range(24):
        fixture.zero_grad(set_to_none=True);native.grad=None
        tick=time.time();loss=compiled(native);torch.neuron.synchronize();forward=time.time()-tick
        tick=time.time();loss.backward();torch.neuron.synchronize();backward=time.time()-tick
        if iteration==0:
            cpu=Fixture(case,reference=True);x=cpu.input().requires_grad_();expected=cpu(x).square().mean();expected.backward()
            actual={'input':native.grad.cpu(),**{n:v.grad.cpu() for n,v in fixture.named_parameters() if v.grad is not None}}
            reference={'input':x.grad,**{n:v.grad for n,v in cpu.named_parameters() if v.grad is not None}}
            if set(actual)!=set(reference):raise AssertionError('Gradient connectivity differs')
            check=compare_gradients(actual,reference);error=abs(float(loss.cpu())-float(expected.detach()))
            if check['relative_l2']>.002 or error>.0002:raise AssertionError('Independent operator numerical gate')
            del cpu,x,actual,reference,expected
        samples.append(dict(iteration=iteration,forward_seconds=forward,backward_seconds=backward))
    report=dict(case=CASES[case],rank=rank,world=world,gradient_relative_l2=check['relative_l2'],
                forward_mean_seconds=statistics.mean(r['forward_seconds'] for r in samples[4:]),
                backward_mean_seconds=statistics.mean(r['backward_seconds'] for r in samples[4:]),
                samples=samples,wall_seconds=time.time()-started,fallback_ops=list(torch_neuronx.get_fallback_ops()))
    (a.output/f'rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n');dist.barrier()
    if rank==0:
        rows=[json.loads((a.output/f'rank{r}.json').read_text()) for r in range(world)]
        if any(set(r['fallback_ops'])-{'aten::_to_copy'} for r in rows):raise AssertionError('Unexpected operator fallback')
        groups={label:[r for r in rows if r['case']==label] for label in CASES}
        result=dict(status='COMPLETE_BLOCK_EXECUTION_PROBE',world=world,research_examples=0,optimizer_updates=0,
                    actual_backward_calls=24*64,wall_seconds=time.time()-started,
                    cases={label:dict(replicas=len(rows),forward_median_seconds=statistics.median(r['forward_mean_seconds'] for r in rows),
                          backward_median_seconds=statistics.median(r['backward_mean_seconds'] for r in rows),
                          maximum_gradient_relative_l2=max(r['gradient_relative_l2'] for r in rows)) for label,rows in groups.items()},
                    scope='Numerical block,rotary,FFN,RMS,interleave and strided-head fixtures across eight rank groups; concurrent-chip conditions are descriptive. No model training,quality result or automatic implementation adoption.')
        (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','plan','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);p.add_argument('--comm-port',type=int,required=True);p.add_argument('--corpus');main(p.parse_args())
