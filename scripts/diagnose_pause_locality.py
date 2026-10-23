"""Exploratory CPU readout of frozen pause checkpoints. No fitting or selection."""
import argparse
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from compute_trainium.data import document_windows, layout, pad_layout
from compute_trainium.metrics import real_token_losses, DocumentScore, summarize
from compute_trainium.recurrence import build_from_config, config_from_signature
from compute_trainium.training import file_sha256
from scripts.qualify_native import digest

BIN_NAMES = ['pre_first_block', 'offset_0', 'offset_1_7', 'offset_8_31', 'offset_32_63', 'offset_64_127']


def bins(context):
    j = torch.arange(1, context+1)
    offset = j % 128
    return torch.where(j < 128, 0, torch.where(offset == 0, 1,
           torch.where(offset < 8, 2, torch.where(offset < 32, 3,
           torch.where(offset < 64, 4, 5)))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--original-root', type=Path, required=True)
    p.add_argument('--threads', type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    spec = {'scope': 'Exploratory, not a new primary endpoint', 'bins': BIN_NAMES,
            'indexing': 'One-based original target j within each 1024-target window; j<128 before first block, else j mod 128',
            'cohort': 'curve', 'seeds': [23, 47, 71], 'endpoint_update': 3840,
            'precision': 'CPU BF16 autocast with FP32 normalized CE',
            'corpus_sha256': file_sha256(args.corpus), 'started_unix': time.time()}
    (args.output/'specification.json').write_text(json.dumps(spec, indent=2)+'\n')
    docs = sorted([d for d in json.loads(args.corpus.read_text()) if d['split']=='dev' and d['cohort']=='curve'], key=lambda d:d['id'])
    index = bins(1024)
    records = []
    for seed in spec['seeds']:
        for arm in ['ordinary', 'early']:
            name = f'natural-curve-v1-{arm}-s{seed}-u3840'
            saved = torch.load(args.original_root/'runs'/name/'checkpoint.pt', map_location='cpu', mmap=True, weights_only=False)
            c = config_from_signature(saved['signature'])
            model = build_from_config(c).eval()
            model.load_state_dict(saved['model'])
            before = digest(model.state_dict())
            pauses = saved['pauses']; padding = saved['signature'].get('pad_to', 0)
            del saved
            scores = []
            with torch.inference_mode():
                for k, doc in enumerate(docs):
                    sums = torch.zeros(6, dtype=torch.float64)
                    counts = torch.zeros(6, dtype=torch.int64)
                    for source, mask in document_windows(doc['tokens'], 1024):
                        b = pad_layout(layout(source[None], mask[None], pauses=pauses), padding)
                        with torch.autocast('cpu', dtype=torch.bfloat16):
                            logits = model(b.inputs, prediction_positions=b.prediction_positions)
                        losses = real_token_losses(logits, b.targets, b.target_mask)[0].double()
                        for i in range(6):
                            chosen = (index == i) & mask
                            sums[i] += losses[chosen].sum()
                            counts[i] += chosen.sum()
                        assert int(counts.sum()) <= len(doc['tokens'])
                    assert int(counts.sum()) == len(doc['tokens'])
                    scores.append({'domain': doc['domain'], 'document': doc['id'], 'group': doc['group'],
                                   'nll_sum_by_bin': sums.tolist(), 'tokens_by_bin': counts.tolist()})
                    if (k+1)%50 == 0: print(json.dumps({'model': name, 'documents': k+1, 'seconds': time.monotonic()-started}), flush=True)
            assert digest(model.state_dict()) == before
            (args.output/(name+'-scores.json')).write_text(json.dumps(scores)+'\n')
            aggregates = [DocumentScore(s['domain'], s['document'], s['group'], sum(s['nll_sum_by_bin']), sum(s['tokens_by_bin'])) for s in scores]
            original = json.loads((args.original_root/'runs'/(name+'-dev')/'report.json').read_text())
            metrics = summarize(aggregates)
            records.append({'seed': seed, 'arm': arm, 'metrics': metrics,
                            'cpu_minus_original_native_weighted_nll': metrics['weighted_nll']-original['metrics']['weighted_nll'],
                            'weights_unchanged': True})
            (args.output/'progress.json').write_text(json.dumps(records, indent=2)+'\n')
            del model, scores
            gc.collect()
    (args.output/'report.json').write_text(json.dumps({'status': 'COMPLETE_PAUSE_LOCALITY_CPU',
        'models': records, 'wall_seconds': time.monotonic()-started, 'source_specification': spec}, indent=2)+'\n')


if __name__=='__main__': main()
