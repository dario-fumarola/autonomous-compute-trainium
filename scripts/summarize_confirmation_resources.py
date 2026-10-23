"""Executed-path cost reconstruction, separate from estimated matmul matching."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    old=a.root.parent;load=lambda path:json.loads(path.read_text())
    follow=load(a.root/'confirmation-campaign-v3/followups/report.json')
    if follow['status']!='COMPLETE_CONFIRMATION_FOLLOWUPS':raise ValueError('All fixed follow-ups must finish')
    def phase(root,name):
        launch=load(root/'snapshots'/name/'launch.json');closure=load(root/'snapshots'/name/'closure.json')
        train=load(root/'runs'/name/'report.json')
        if closure['exit_code']!=0 or train['status']!='PASS_PACKED_TRAINING_PHASE':raise ValueError('Invalid retained training phase')
        return {'run':name,'seconds':closure['wall_seconds'],'chips':len(launch['physical_chips']),
            'warm_real_targets_per_second':train['warm_global_real_targets_per_second'],
            'method':'Full training-process launch/closure, including compile, recovery and checkpoint hashing'},train
    ordinary=[]
    for update in [384,1152]:ordinary.append(phase(old,f'natural-curve-v1-ordinary-s23-u{update}')[0])
    name='natural-curve-v1-ordinary-s23-u3840'
    launch=load(old/'snapshots'/name/'launch.json');closure=load(old/'snapshots'/name/'closure.json')
    parent=old/'runs'/name/'stable-u3456.pt'
    prefix_seconds=parent.stat().st_mtime-launch['started_unix']
    if not 0<prefix_seconds<closure['wall_seconds']:raise ValueError('Historical parent timestamp not inside its training process')
    ordinary.append({'run':name+' (retained prefix to3456)','seconds':prefix_seconds,
        'chips':len(launch['physical_chips']),
        'method':'Prefix ends at original stable-checkpoint filesystem mtime. Includes writing that state; excludes later annealed suffix and terminal teardown. Timestamp reconstruction, not a new timing experiment.'})
    terminal=None
    for update in [5846,6496]:
        row,terminal=phase(a.root,f'confirmation-v3-ordinary-flop-s23-u{update}');ordinary.append(row)
    result=[]
    def summarize(identity,parts,train):
        return {'identity':identity,'parameters':train['parameters'],'real_targets':train['total_real_targets'],
            'training_matmul_flops':train['total_training_matmul_flops'],
            'forward_matmul_flops_per_1024_window':train['flops_per_global_step']['forward_matmul_flops']/256,
            'training_process_seconds':sum(p['seconds'] for p in parts),
            'assigned_training_chip_hours':sum(p['seconds']*p['chips']/3600 for p in parts),
            'training_process_starts':len(parts),'retained_path':parts}
    result.append(summarize('ordinary-flop-s23',ordinary,terminal))
    for seed in [23,47,71]:
        prefix='untied-depth-v1' if seed==23 else 'confirmation-v3-untied'
        parts=[]
        for update in [3456,3840]:
            row,terminal=phase(a.root,f'{prefix}-s{seed}-u{update}');parts.append(row)
        result.append(summarize(f'untied-1b-s{seed}',parts,terminal))
    report={'status':'COMPLETE_RETAINED_PATH_RESOURCE_LEDGER','models':result,
        'flop_match':follow['specification']['flop_match'],
        'scope':'Training-process costs along the retained paths, including reused historical ordinary states; not new campaign charges. Separate incremental execution-costs.json counts all qualification, tuning, readouts and failed attempts. DEV evaluation, CPU audits and idle/controller gaps are not in these per-model training sums. Ordinary uses five process segments versus two for untied; its retained-parent segment uses filesystem timestamp reconstruction and omits that segment final teardown. Warm throughput and the full observed campaign span are therefore reported separately. Dense FLOPs exclude optimizer, elementwise, communication and compilation; inference forward FLOPs are estimates, not generation latency.'}
    (a.output/'resource-costs.json').write_text(json.dumps(report,indent=2)+'\n')
    text='''# Resource comparison

This records the executed retained training paths, including the ordinary
model's reused pre-annealing history. Estimated matmul matching and measured
execution costs are different quantities.

| Model | Parameters | Real targets | Matmul FLOPs | Training process minutes | Assigned training chip-hours | Process segments |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
'''
    for r in result:text+=f"| {r['identity']} | {r['parameters']:,} | {r['real_targets']:,} | {r['training_matmul_flops']:.6e} | {r['training_process_seconds']/60:.2f} | {r['assigned_training_chip_hours']:.3f} | {r['training_process_starts']} |\n"
    text+='''
The ordinary retained prefix ends at its original stable-checkpoint file
timestamp; this reconstructs historical elapsed time and omits that prefix's
terminal teardown. Its five process segments versus two for untied include
different compile/recovery overhead. These sums exclude DEV scoring, CPU audits,
tuning, qualification and idle gaps. The separate campaign execution ledger
retains all of those attempts and reports both accelerator-active time and the
whole observed launch span. Reused controls are not charged a second time to
incremental campaign cost. No claim about total efficiency follows from the
dense-FLOP match alone.

The two-pass path executes about 1.692 times the ordinary forward matmul work
per fixed-context scoring window. These estimates include the vocabulary
projection; they are not autoregressive generation latency measurements.

[Detailed resource ledger](resource-costs.json) · [Learning comparison](REPORT.md).
'''
    (a.output/'RESOURCES.md').write_text(text);print(json.dumps({'models':len(result),'status':report['status']}))


if __name__=='__main__':main()
