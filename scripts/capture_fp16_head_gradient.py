"""Native FP16-body/FP32-head scaled gradients; no optimizer or updates."""
import argparse
import contextlib
import ctypes
import gc
import json
import os
import time
from pathlib import Path

os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA'] = '0'
os.environ['NEURON_RT_VISIBLE_CORES'] = os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK', '0'))]

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from compute_trainium.data import prediction_spans
from compute_trainium.metrics import real_token_losses
from compute_trainium.depth_mixture import routed_indices
from compute_trainium.preln_recurrence import PreNormRecurrenceConfig, build_preln_recurrent
from compute_trainium.training import PackedCorpus, atomic_checkpoint, file_sha256
from scripts.qualify_native import digest


class Objective(torch.nn.Module):
    def __init__(self, model, spans):
        super().__init__()
        self.model, self.spans = model, spans

    def forward(self, inputs, targets, mask):
        with torch.autocast('neuron', dtype=torch.float16):
            h = self.model.hidden(inputs)
        with torch.autocast('neuron', enabled=False):
            h = torch.cat([h[:, a:b] for a, b in self.spans], dim=1)
            logits = F.linear(h.float(), self.model.decoder.lm_head.weight)
            return real_token_losses(logits, targets, mask).sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--expected-checkpoint-sha256', required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--packed-corpus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import torch_neuronx
    torch.set_num_threads(2)
    torch.neuron.set_device(0)
    dist.init_process_group('neuron')
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 32: raise ValueError('Registered half-node32-rank capture only')
    a.output.mkdir(exist_ok=True)
    started = time.time()
    if rank == 0 and file_sha256(a.checkpoint) != a.expected_checkpoint_sha256:
        raise AssertionError('Registered mature source changed')
    saved = torch.load(a.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    sig, cursor, update = saved['signature'], saved['cursor'], saved['update']
    if (sig['world'], sig['batch'], sig['accumulation']) != (32, 4, 2) or saved['pauses']:
        raise ValueError('Preserved32x4x2 source required; capture reroutes same256 windows')
    config = PreNormRecurrenceConfig(**sig['model'])
    model = build_preln_recurrent(config)
    model.load_state_dict(saved['model'])
    before = digest(model.state_dict())
    del saved
    if file_sha256(a.corpus) != sig['corpus_sha256']: raise AssertionError('Source changed')
    packed = PackedCorpus.from_mapped(a.packed_corpus, 1024, sig['data_seed'], sig['corpus_sha256'])
    if packed.fingerprint != sig['packing_sha256']: raise AssertionError('Packing changed')
    count = int((packed.prefix_counts[cursor+256]-packed.prefix_counts[cursor]).sum())
    def batch(micro):
        return packed.batch(routed_indices(cursor, micro, rank, 32, 1, 8, sig['routing_seed']), 0)[0]
    def gradients():
        result = {}
        for name, value in model.named_parameters():
            if value.grad.dtype != torch.float32: raise AssertionError('Expected FP32 master gradient')
            value.grad.div_(8192.)
            result[name] = value.grad.detach().cpu().clone()
        if not all(torch.isfinite(g).all() for g in result.values()):
            raise AssertionError('Nonfinite unscaled gradient')
        return result
    first = batch(0)
    model.to('neuron')
    gc.collect()
    ctypes.CDLL(None).malloc_trim(0)
    objective = Objective(model, prediction_spans(first.prediction_positions))
    torch._dynamo.config.optimize_ddp = False
    compiled = torch.compile(objective, backend='neuron', fullgraph=True, dynamic=False)
    own = batch(0)
    local = compiled(*(t.contiguous().to('neuron') for t in [own.inputs, own.targets, own.target_mask]))/own.target_mask.sum().item()
    (local*8192).backward()
    local_grads = gradients()
    if rank == 0:
        atomic_checkpoint(a.output/'local-native.pt', {'gradients': local_grads, 'loss': float(local.detach().cpu())})
    del local, local_grads
    model.zero_grad(set_to_none=True)
    ddp = DDP(compiled, device_ids=None, broadcast_buffers=False, bucket_cap_mb=64, gradient_as_bucket_view=True)
    total = torch.zeros((), device='neuron')
    for micro in range(8):
        own = batch(micro)
        with ddp.no_sync() if micro < 7 else contextlib.nullcontext():
            loss = ddp(*(t.contiguous().to('neuron') for t in [own.inputs, own.targets, own.target_mask]))
            (loss*(32/count)*8192).backward()
        total += loss.detach()
    dist.all_reduce(total)
    values = gradients()
    after = digest(model.state_dict())
    if before != after: raise AssertionError('Gradient capture changed model weights')
    own_digest = digest(values)
    (a.output/f'replica-{rank}.json').write_text(json.dumps({'gradient_digest': own_digest,
        'weight_digest': after, 'fallback_ops': list(torch_neuronx.get_fallback_ops())})+'\n')
    dist.barrier()
    if rank == 0:
        replicas = [json.loads((a.output/f'replica-{i}.json').read_text()) for i in range(32)]
        if len({r['gradient_digest'] for r in replicas}) != 1 or len({r['weight_digest'] for r in replicas}) != 1:
            raise AssertionError('Replica gradients/weights differ')
        if any(set(r['fallback_ops'])-{'aten::_to_copy'} for r in replicas):
            raise AssertionError('Unexpected native fallback')
        if file_sha256(a.checkpoint) != a.expected_checkpoint_sha256: raise AssertionError('Source file changed')
        atomic_checkpoint(a.output/'global-native.pt', {'gradients': values, 'loss': float(total.cpu())/count})
        report = {'status': 'CAPTURED_NATIVE_FP16_HEAD_FP32_SCALED_GLOBAL_GRADIENT',
            'checkpoint_sha256': a.expected_checkpoint_sha256, 'source_update': update, 'real_targets': count,
            'source_cursor': cursor, 'loss_scale': 8192, 'capture_batch': 1, 'capture_accumulation': 8,
            'native_nll': float(total.cpu())/count, 'all_replica_gradients_equal': True,
            'weights_unchanged': True, 'optimizer_updates': 0, 'test_accessed': False,
            'wall_seconds': time.time()-started, 'gradient_digest': own_digest,
            'scope': 'FP16 body, FP32 vocabulary/CE, explicit fixed8192 scaling and FP32 unscale. Full256-window global capture only; acceptance against preserved CPU FP32 reference is separate. No optimizer/learning/throughput qualification.'}
        (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
