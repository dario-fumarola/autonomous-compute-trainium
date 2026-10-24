"""Bounded FP16-body feasibility on existing natural TRAIN numerical fixtures."""
import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_fp16_candidate import ParcaeFP16Candidate
from compute_trainium.parcae_stages_candidate import ParcaeStagesCandidate
from compute_trainium.parcae_update_qualification import relative_error
from compute_trainium.queue import write_state
from compute_trainium.training import PackedCorpus, file_sha256


def group(name):
    if name == 'transformer.wte.weight':
        return 'tied_input_output'
    if name.startswith('value_embeds.'):
        return 'value_embeddings'
    if name.startswith('transformer.adapter.') or name.startswith('transformer.C.'):
        return 'recurrent_injection_readout'
    return 'transformer_body'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    out = args.root / 'parcae-cpu-fp16-v1'
    public = Path('results/parcae_cpu_fp16_20261024')
    out.mkdir(exist_ok=False); public.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(2)
    reference = args.root / 'parcae-native-global-v1'
    proof = json.loads((reference / 'report.json').read_text())
    source_plan = json.loads((reference / 'plan.json').read_text())
    files = ['scripts/screen_parcae_fp16_cpu.py', 'compute_trainium/parcae_fp16_candidate.py',
        'compute_trainium/parcae_stages_candidate.py', 'compute_trainium/parcae_candidate.py']
    plan = {'identity': 'parcae-cpu-fp16-v1', 'registered_unix': started,
        'source_sha256': {f: file_sha256(f) for f in files},
        'cpu_reference_report_sha256': file_sha256(reference / 'report.json'),
        'reference_plan_sha256': file_sha256(reference / 'plan.json'),
        'training_windows': 8, 'scale': 8192, 'gradient_relative_gate': .02,
        'nll_absolute_gate': .003, 'cpu_seconds_limit': 600, 'assigned_chips': 0,
        'optimizer_updates': 0, 'test_accessed': False,
        'scope': 'Existing natural-TRAIN numerical fixtures at perturbed qualification weights. FP16 body with FP32 final norm, vocabulary projection, masters, gradients and CE. Manual loss scale8192. CPU feasibility only, no native execution or learning adoption.'}
    write_state(out / 'plan.json', plan); write_state(public / 'plan.json', plan)
    records = []
    result = None

    def guard():
        if time.time()-started > 600:
            raise TimeoutError('CPU FP16 ten-minute preparation cap')
        available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        if available < 544*1024**3 or min(shutil.disk_usage(out).free, shutil.disk_usage('/').free) < 35*1024**3:
            raise RuntimeError('CPU FP16 host reserve guard')

    try:
        guard()
        if proof['status'] != 'COMPLETE_INDEPENDENT_PARCAE_CPU_GLOBAL_REFERENCES':
            raise ValueError('Independent CPU reference preparation is incomplete')
        packed = PackedCorpus.from_mapped(args.root / 'natural-append-6500m-v2/packed',
            1024, 23, source_plan['source_corpus_sha256'])
        if packed.fingerprint != source_plan['packing_fingerprint']:
            raise AssertionError('Natural TRAIN packing changed')
        if file_sha256(reference / 'initial-state.npy') != source_plan['initial_state_file_sha256']:
            raise AssertionError('Qualification initial states changed')
        noise = np.load(reference / 'initial-state.npy', mmap_mode='r')
        total = sum(packed.counts(0, 8))
        for row in proof['records']:
            begin = time.time()
            arm = row['arm']
            checkpoint = reference / arm / 'qualification-state.pt'
            if file_sha256(checkpoint) != row['checkpoint_sha256']:
                raise AssertionError('Qualification weights changed')
            saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
            if not saved['qualification_only']:
                raise ValueError('Numerical qualification point required')
            config = ParcaeCandidateConfig(**saved['config'])
            vectors, losses = [], []
            for mixed in [False, True]:
                cls = ParcaeFP16Candidate if mixed else ParcaeStagesCandidate
                model = cls(config, recurrent=saved['recurrent'], initialize=False)
                model.load_state_dict(saved['model'])
                loss_total = 0.
                scale = 8192 if mixed else 1
                with sdpa_kernel(SDPBackend.MATH):
                    for i in range(8):
                        guard()
                        batch = packed.batch([i], 0)[0]
                        state = torch.from_numpy(np.array(noise[i:i+1], copy=True)) if saved['recurrent'] else None
                        kwargs = {'total_steps': 4, 'backward_cap': 2} if saved['recurrent'] else {}
                        logits = model(batch.inputs, state, **kwargs)
                        if logits.dtype != torch.float32:
                            raise AssertionError('Expected FP32 head and CE')
                        value = (F.cross_entropy(logits.flatten(0,1), batch.targets.flatten(), reduction='none')*batch.target_mask.flatten()).sum()
                        (value * (scale/total)).backward()
                        loss_total += float(value.detach())
                gradients = {n: p.grad.detach().clone()/scale for n,p in model.named_parameters()}
                if not all(p.dtype == torch.float32 for p in model.parameters()):
                    raise AssertionError('Master dtype changed')
                vectors.append(gradients); losses.append(loss_total/total)
                del model
                gc.collect()
            by_group = {}
            for category in sorted({group(n) for n in vectors[0]}):
                names = [n for n in vectors[0] if group(n) == category]
                by_group[category] = relative_error({n: vectors[0][n] for n in names}, {n: vectors[1][n] for n in names})
            comparison = relative_error(vectors[0], vectors[1])
            passes = comparison['relative_l2'] <= .02 and abs(losses[1]-losses[0]) <= .003 and all(r['relative_l2'] <= .02 for r in by_group.values())
            records.append({'arm': arm, 'real_targets': total, 'gradient_comparison': comparison,
                'gradient_groups': by_group, 'fp32_nll': losses[0], 'fp16_body_nll': losses[1],
                'passes_registered_screen': passes, 'active_seconds': time.time()-begin})
            write_state(public / 'progress.json', {'records': records})
            del saved, vectors, gradients
            gc.collect()
        if any(file_sha256(f) != sha for f,sha in plan['source_sha256'].items()):
            raise AssertionError('CPU mixed-precision candidate source changed')
        result = {'status': 'PASS_CPU_PARCAE_FP16_BODY_SCREEN' if all(r['passes_registered_screen'] for r in records) else 'FAILED_CPU_PARCAE_FP16_BODY_SCREEN',
            'records': records, 'optimizer_updates': 0, 'assigned_chips': 0,
            'test_accessed': False, 'wall_seconds_after_imports': time.time()-started, 'scope': plan['scope']}
    except BaseException as error:
        result = {'status': 'STOPPED_CPU_PARCAE_FP16_BODY_SCREEN', 'error': str(error),
            'records': records, 'assigned_chips': 0, 'optimizer_updates': 0,
            'test_accessed': False, 'wall_seconds_after_imports': time.time()-started}
        raise
    finally:
        write_state(out / 'report.json', result); write_state(public / 'report.json', result)


if __name__ == '__main__':
    main()
