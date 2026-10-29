"""Describe existing TRAIN audit gradients; no new model evaluation."""
import argparse
import json
from pathlib import Path
import time

from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def run(root):
    import torch
    torch.set_num_threads(8)
    start = time.time()
    out = root/'depth-causal-gradient-alignment-v1'
    public = Path('results/depth_causal_workspace_20261029/gradient-alignment-v1')
    if out.exists() or public.exists():
        raise FileExistsError('Preserve preceding analysis')
    audits, paths = {}, {}
    for i in range(3):
        a = root/f'depth-causal-tuning-v1/audit-depth-lr{i}/report.json'
        if json.loads(a.read_text())['status'] != 'PASS_INTERLEAVED_SCREEN_MATURE_AUDITS':
            raise AssertionError('All audited endpoints required')
        audits[str(i)] = file_sha256(a)
        for k in range(3):
            paths[i, k] = root/f'runs/depth-causal-tuning-v1-depth-lr{i}/PRIVATE-gradient-u1957-k{k}.pt'
    plan = dict(registered_unix=start, seconds_limit=120, source_sha256=file_sha256(__file__),
        audit_sha256=audits, captures={f'lr{i}-b{k}': file_sha256(p) for (i,k),p in paths.items()},
        pairs=[[0,1],[1,2],[0,2]], accumulator='FP64 dot products and squared norms',
        scope='Descriptive alignment on the same previously audited TRAIN batch at all three endpoints. No new model forward/backward, data, held-out read, optimizer change, LR selection or advancement decision. Raw gradients are not AdamW updates or a causal explanation.')
    out.mkdir(mode=0o700)
    public.mkdir(parents=True)
    write_state(out/'plan.json', plan)
    write_state(public/'plan.json', plan)
    rows = []
    for i in range(3):
        captured = {k: torch.load(paths[i,k], map_location='cpu', weights_only=True, mmap=True) for k in range(3)}
        if len({(v['count'], v['window_start']) for v in captured.values()}) != 1:
            raise AssertionError('Exact shared TRAIN batch required')
        for left, right in plan['pairs']:
            a, b = captured[left]['gradients'], captured[right]['gradients']
            if a.keys() != b.keys():
                raise AssertionError('Same parameter inventory required')
            sums = {key:[0.,0.,0.] for key in ('all','embedding','readout','body')}
            for name in a:
                x,y = a[name].double(),b[name].double()
                terms = (float((x*y).sum()),float(x.square().sum()),float(y.square().sum()))
                family = 'embedding' if name == 'embedding.weight' else 'readout' if name == 'prediction_weight' else 'body'
                for key in ('all',family):
                    sums[key] = [u+v for u,v in zip(sums[key],terms)]
            statistics = {}
            for name,(dot,aa,bb) in sums.items():
                cosine = dot/(aa*bb)**.5
                if not aa > 0 or not bb > 0 or not -1.00000001 <= cosine <= 1.00000001:
                    raise AssertionError('Finite nonempty gradient families required')
                statistics[name] = dict(cosine=cosine, left_norm=aa**.5, right_norm=bb**.5, norm_ratio=(bb/aa)**.5)
            rows.append(dict(lr_index=i,left_budget=left,right_budget=right,families=statistics))
            if time.time()-start > plan['seconds_limit']:
                raise TimeoutError('Finite analysis cap')
        del captured
    result = dict(status='COMPLETE_DEPTH_CAUSAL_TRAIN_GRADIENT_ALIGNMENT', rows=rows,
        wall_seconds=time.time()-start, assigned_chips=0, new_model_passes=0, heldout_reads=0,
        optimizer_updates=0, scope=plan['scope'])
    write_state(out/'report.json', result)
    write_state(public/'report.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
