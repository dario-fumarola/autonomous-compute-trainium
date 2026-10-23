"""Optimizer software fixtures: FP32 AdamW, offset gradient views and recovery.

No research examples or language-model fits. Run on a free core, separately
from scientific training. This does not measure whole-model throughput.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import torch

from compute_trainium.training import cpu_tree
from scripts.qualify_native import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--corpus');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['plain','foreach','fused'],default='fused');args=p.parse_args()
    torch.set_num_threads(4);torch.neuron.set_device(0);torch.manual_seed(29)
    shapes=[(128,128),(768,),(1024,256),(33,)]
    original=[torch.randn(shape,dtype=torch.float32)*.02 for shape in shapes]
    cpu=[torch.nn.Parameter(x.clone()) for x in original]
    device=[torch.nn.Parameter(x.clone().to('neuron')) for x in original]
    cpu_opt=torch.optim.AdamW(cpu,lr=.001,foreach=False,fused=False)
    opt=torch.optim.AdamW(device,lr=.001,foreach=args.mode=='foreach',fused=args.mode=='fused')
    offsets=[];n=37
    for x in original:offsets.append(n);n+=x.numel()+19
    grad_cpu=torch.randn(n,dtype=torch.float32)*.01;grad_device=grad_cpu.to('neuron')
    maximum_error=0.
    def step(parameters,optimizer,buffer,index):
        optimizer.zero_grad(set_to_none=True)
        buffer.mul_(.95)
        for param,offset in zip(parameters,offsets,strict=True):
            param.grad=buffer[offset:offset+param.numel()].view_as(param)
        for group in optimizer.param_groups:group['lr']=[.001,.0003,.002][index%3]
        optimizer.step()
    def compare():
        nonlocal maximum_error
        for a,b in zip(device,cpu,strict=True):
            actual=a.detach().cpu();maximum_error=max(maximum_error,float((actual-b.detach()).abs().max()))
            torch.testing.assert_close(actual,b.detach(),atol=2e-6,rtol=1e-4)
            for key in ['step','exp_avg','exp_avg_sq']:
                aa=opt.state[a][key].detach().cpu();bb=cpu_opt.state[b][key]
                if key=='step':assert torch.equal(aa,bb)
                else:
                    assert aa.dtype==torch.float32
                    torch.testing.assert_close(aa,bb,atol=1e-10 if key=="exp_avg_sq" else 2e-8,rtol=2e-4)
    args.output.mkdir(parents=True,exist_ok=True)
    for i in range(12):
        step(cpu,cpu_opt,grad_cpu,i);step(device,opt,grad_device,i);compare()
    saved={'parameters':[x.detach().cpu() for x in device],'optimizer':cpu_tree(opt.state_dict()),
           'grad_buffer':grad_device.cpu()}
    torch.save(saved,args.output/'optimizer-fixture.pt')
    for i in range(12,15):
        step(cpu,cpu_opt,grad_cpu,i);step(device,opt,grad_device,i);compare()
    expected=digest({'parameters':device,'optimizer':opt.state_dict()})
    loaded=torch.load(args.output/'optimizer-fixture.pt',weights_only=False)
    restored=[torch.nn.Parameter(x.to('neuron')) for x in loaded['parameters']]
    recovered=torch.optim.AdamW(restored,lr=.001,foreach=args.mode=='foreach',fused=args.mode=='fused')
    recovered.load_state_dict(loaded['optimizer']);buffer=loaded['grad_buffer'].to('neuron')
    for i in range(12,15):step(restored,recovered,buffer,i)
    actual=digest({'parameters':restored,'optimizer':recovered.state_dict()})
    if actual!=expected:raise AssertionError('Full fused optimizer continuation differs')
    torch.neuron.synchronize();before=time.monotonic()
    for i in range(100):step(restored,recovered,buffer,i)
    torch.neuron.synchronize();seconds=time.monotonic()-before
    import torch_neuronx
    report={'status':'PASS_OPTIMIZER_SOFTWARE_FIXTURE','mode':args.mode,'parameters':sum(x.numel() for x in original),
            'maximum_cpu_parameter_error':maximum_error,'fp32_moments':True,'offset_gradient_bucket_views':True,
            'changing_lr_cpu_agreement':True,'full_optimizer_continuation_exact':True,
            'warm_optimizer_only_updates':100,'warm_seconds':seconds,'fallback_ops':str(torch_neuronx.get_fallback_ops()),
            'scope':'Numerical software fixture, not a corpus or whole-model performance estimate'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
