"""Public, content-free SDPA precision reproducer for the native SDK.

Eight independent processes pair NKI/primitive attention at four input scales.
No checkpoint, token data, optimizer or research examples are used. Launch with
eight explicit visible cores through run_native.sh; no collectives are needed.
"""
import argparse
import json
import os
import time
from pathlib import Path

rank = int(os.environ.get('LOCAL_RANK', '0'))
os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA'] = str(rank % 2)
# Different lowering policies must never reuse an earlier policy's cached HLO.
os.environ['TORCH_NEURONX_HLO_COMPILE_CACHE'] = '0'
os.environ['TORCH_NEURONX_HLO_CACHE'] = '0'
os.environ['NEURON_RT_VISIBLE_CORES'] = os.environ['RESEARCH_VISIBLE_CORES'].split(',')[rank]

import torch
from torch.nn import functional as F
from scripts.qualify_trained_gradient import compare_gradients


def objective(upstream, q, k, v):
    output = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    return (output * upstream).sum(), output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus')  # Standard launcher; no corpus opened.
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--deadline-unix', type=float, required=True)
    args = p.parse_args()
    args.output.mkdir(exist_ok=True)
    if int(os.environ.get('WORLD_SIZE', '1')) != 8:
        raise ValueError('Eight paired cases required')
    started = time.time()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(87319)
    scale = [1, 2, 4, 8][rank // 2]
    xs = [torch.randn(1, h, 1024, 64) * (scale if i < 2 else 1)
          for i, h in enumerate([12, 4, 4])]
    upstream = torch.randn(1, 12, 1024, 64) / (12*1024*64)**.5
    refs = {}
    for dtype in [torch.float32, torch.float64]:
        current = [x.to(dtype).clone().requires_grad_() for x in xs]
        loss, output = objective(upstream.to(dtype), *current)
        loss.backward()
        refs[str(dtype)] = {'loss': float(loss.detach()), 'output': output.detach().float(),
                            'gradients': {str(i): x.grad.float() for i, x in enumerate(current)}}
    import torch_neuronx
    torch.neuron.set_device(0)
    current = [x.to('neuron').requires_grad_() for x in xs]
    compiled = torch.compile(objective, backend='neuron', fullgraph=True, dynamic=False)
    loss, output = compiled(upstream.to('neuron'), *current)
    loss.backward()
    native = {str(i): x.grad.cpu() for i, x in enumerate(current)}
    y = output.detach().cpu()
    metrics = {label: {'gradient': compare_gradients(native, ref['gradients']),
        'output_relative_l2': float((y-ref['output']).norm()/ref['output'].norm()),
        'loss_difference': float(loss.detach().cpu())-ref['loss']} for label, ref in refs.items()}
    row = {'qk_input_standard_deviation': scale, 'nki_sdpa_enabled': bool(rank % 2),
           'dtype': 'float32', 'matmul_precision': torch.get_float32_matmul_precision(),
           'hlo_caches_disabled': True, 'metrics': metrics, 'wall_seconds': time.time()-started,
           'fallback_ops': sorted(map(str, torch_neuronx.get_fallback_ops()))}
    (args.output/f'case-{rank}.json').write_text(json.dumps(row, indent=2)+'\n')
    if rank == 0:
        while not all((args.output/f'case-{i}.json').exists() for i in range(8)):
            if time.time() > args.deadline_unix:
                raise TimeoutError('SDPA reproducer deadline')
            time.sleep(1)
        (args.output/'report.json').write_text(json.dumps({'status': 'COMPLETE_FP32_SDPA_PRECISION_REPRODUCER',
            'cases': [json.loads((args.output/f'case-{i}.json').read_text()) for i in range(8)],
            'wall_seconds': time.time()-started,
            'scope': 'Numerical fixtures only. No private source data, model weights, optimizer steps, training result or TEST.'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
