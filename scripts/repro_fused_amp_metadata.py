"""Operator-contract fixture: optional fused-AdamW unscale/overflow inputs.

No corpus or scientific-model updates. A CPU operator on the SAME PyTorch
version is the reference. High-level GradScaler reachability is not tested.
"""
import argparse,importlib.metadata,json
from pathlib import Path
import torch


def call(device,scale,overflow):
    parameter=torch.full((128,128),.02,device=device)
    gradient=torch.full_like(parameter,.01)
    moment=torch.full_like(parameter,.2);variance=torch.full_like(parameter,.4)
    step=torch.tensor(1.,device=device)
    grad_scale=torch.tensor(float(scale),device=device) if scale is not None else None
    found_inf=torch.tensor(float(overflow),device=device) if overflow is not None else None
    torch._fused_adamw_([parameter],[gradient],[moment],[variance],[],[step],lr=.001,
        beta1=.9,beta2=.999,weight_decay=.01,eps=1e-8,amsgrad=False,maximize=False,
        grad_scale=grad_scale,found_inf=found_inf)
    return [x.detach().cpu() for x in [parameter,moment,variance]]


def main():
    p=argparse.ArgumentParser();p.add_argument('--corpus');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    cases=[('normal',None,None),('overflow_skip',None,1.),('gradient_unscale',4.,0.)]
    try:references={name:call('cpu',scale,overflow) for name,scale,overflow in cases}
    except (RuntimeError,NotImplementedError) as e:
        report={'status':'CPU_OPERATOR_CONTRACT_UNAVAILABLE','reason':str(e),'scope':'No native defect inferred'}
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return
    torch.neuron.set_device(0);rows=[]
    for name,scale,overflow in cases:
        expected=references[name];actual=call('neuron',scale,overflow)
        errors={label:float((x-y).abs().max()) for label,x,y in zip(['parameter','exp_avg','exp_avg_sq'],actual,expected,strict=True)}
        passed=all(torch.allclose(x,y,atol=1e-10 if label=='exp_avg_sq' else 2e-8,rtol=2e-5)
            for label,x,y in zip(['parameter','exp_avg','exp_avg_sq'],actual,expected,strict=True))
        rows.append({'case':name,'cpu_agreement':passed,'maximum_errors':errors,
            'cpu_means':[float(x.mean()) for x in expected],'native_means':[float(x.mean()) for x in actual]})
    if not rows[0]['cpu_agreement']:status='NORMAL_FUSED_OPERATOR_MISMATCH'
    else:status='REPRODUCED_FUSED_OPTIONAL_METADATA_MISMATCH' if any(not r['cpu_agreement'] for r in rows[1:]) else 'PASS_FUSED_OPTIONAL_METADATA_CONTRACT'
    report={'status':status,'cases':rows,'torch':importlib.metadata.version('torch'),
        'torch_neuronx':importlib.metadata.version('torch-neuronx'),
        'scope':'Direct ATen operator comparison with same-version CPU. High-level GradScaler support/reachability is untested. Scientific training uses unscaled BF16 with FP32 masters, not these optional metadata arguments.'}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if status not in ['PASS_FUSED_OPTIONAL_METADATA_CONTRACT']:raise SystemExit(1)


if __name__=='__main__':main()
