"""Existing independent DEV readout with exact current state/variant policy."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--checkpoint-sha',required=True);p.add_argument('--case',required=True)
    p.add_argument('--depth',type=int,required=True)
    a,_=p.parse_known_args()
    import torch
    from compute_trainium.training import file_sha256
    from compute_trainium.lean_baseline_science import make_reference,validate_point,ROLES,evaluation_window_key
    if file_sha256(a.plan)!=a.plan_sha or file_sha256(a.checkpoint)!=a.checkpoint_sha:raise AssertionError('Exact baseline readout identity required')
    plan=json.loads(a.plan.read_text())
    if any(file_sha256(f)!=sha for f,sha in plan['source_sha256'].items()):raise AssertionError('Exact baseline readout source changed')
    state=torch.load(a.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    validate_point(state,a.plan_sha,plan,a.case,root=a.root,parent_file=a.plan);del state
    role=plan['model_variant']
    if a.depth!=ROLES[role][1]:raise ValueError('Only the frozen supported baseline operating point may be scored')
    from scripts import evaluate_low_depth_cpu as engine
    def reference(config,recurrent):
        if recurrent!=(role!='ordinary8'):raise ValueError('Readout baseline role differs')
        model=make_reference(config,role)
        if role=='mixing':
            forward=model.forward
            def full_mixing(*args,num_steps_pair=None,**kwargs):
                if num_steps_pair is None:raise ValueError('Explicit two-mix readout required')
                return forward(*args,num_steps_pair=torch.tensor([0,int(num_steps_pair.sum())]),**kwargs)
            model.forward=full_mixing
        return model
    engine.make_reference=reference
    # Keep fixed1024 draws before prefix slicing; no document/content/length key.
    engine.window_state_key=evaluation_window_key
    engine.main()


if __name__=='__main__':main()
