"""Read-only aggregate check: did failed learned-cache parameters actually move?"""
import argparse
import json
import time
from pathlib import Path
import torch
from compute_trainium.causal_associative_workspace import AssociativeWorkspace
from compute_trainium.training import file_sha256


def main(root):
    start = time.time(); folder = root/'causal-associative-workspace-v1'
    plan = json.loads((folder/'plan.json').read_text())
    path = folder/'PRIVATE-learned-state.pt'; before = file_sha256(path)
    torch.manual_seed(plan['seed']); model = AssociativeWorkspace(mode='learned')
    saved = torch.load(path, weights_only=True)
    if saved['updates'] != plan['updates']:
        raise AssertionError('Complete registered endpoint required')
    movement = {}
    for name in ['query.weight', 'key.weight']:
        initial = model.state_dict()[name]; final = saved['model'][name]
        movement[name] = dict(initial_l2=float(initial.norm()), final_l2=float(final.norm()),
                             change_l2=float((final-initial).norm()))
    a = json.loads((folder/'learned-telemetry.json').read_text())
    b = json.loads((folder/'cosine-telemetry.json').read_text())
    if len(a)!=1536 or len(b)!=1536:
        raise AssertionError('Complete telemetry required')
    windows = []
    for low, high in [(0,384), (1152,1536)]:
        windows.append(dict(first_update=low+1, last_update=high,
            learned_minus_cosine_online_loss=sum(a[i]['loss']-b[i]['loss'] for i in range(low,high))/(high-low),
            mean_learned_strength=sum(a[i]['strength'] for i in range(low,high))/(high-low)))
    if file_sha256(path)!=before:
        raise AssertionError('Read-only diagnostic changed state')
    result = dict(optimizer_updates=0, assigned_chips=0, checkpoint_unchanged=True,
        checkpoint_sha256=before, movement=movement, online_training_windows=windows,
        scope='Post-result TRAIN-only inspection,not new held-out evidence. Parameter movement rules out completely absent updates; it does not establish a good optimizer,adequate sample size or learned useful computation.',
        wall_seconds=time.time()-start)
    for dest in [folder/'training-signal-analysis.json',
                 Path('results/causal_associative_workspace_20261028/training-signal-analysis.json')]:
        dest.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
