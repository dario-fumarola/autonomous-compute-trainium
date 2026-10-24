"""Retained model paths versus actual campaign cost; shared ancestors stay explicit."""
import argparse
import json
import time
from pathlib import Path


def model_stages(arm,seed,endpoint):
    updates=[384,1152,3456,3840] if endpoint=='1B' else [384,1152,3456,5846,6496]
    if endpoint!='1B' and arm!='ordinary':raise ValueError('Only ordinary has the matched-work branch')
    return [f'vdepth-v1-{arm}-s{seed}-u{u}' for u in updates]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=[];pending=[];phase_union=set()
    for seed in [103,211,307]:
        for arm,endpoint in [('ordinary','1B'),('ordinary','matched'),('fixed','1B'),('mixture','1B')]:
            names=model_stages(arm,seed,endpoint);reports=[];phases=[]
            if any(not (a.root/'runs'/name/'report.json').exists() or
                   not (a.root/'snapshots'/name/'closure.json').exists() for name in names):
                pending.append({'arm':arm,'seed':seed,'endpoint':endpoint});continue
            parent=None;initial=set();corpus=set();targets=updates=flops=probes=0;wall=warm=assigned=0.
            for name in names:
                report=json.loads((a.root/'runs'/name/'report.json').read_text())
                launch=json.loads((a.root/'snapshots'/name/'launch.json').read_text())
                closure=json.loads((a.root/'snapshots'/name/'closure.json').read_text())
                if closure['exit_code']!=0 or report['status']!='PASS_PACKED_TRAINING_PHASE':
                    raise ValueError('Unsuccessful phase in a retained path: '+name)
                if not report['all_replica_model_optimizer_states_equal']:raise ValueError('Replica state differs')
                args=launch['arguments']
                if parent is not None:
                    if report['recovered_model_optimizer_digest']!=parent['state_digest']:
                        raise ValueError('Path does not inherit the declared full optimizer state')
                    resume=args[args.index('--resume')+1]
                    if Path(resume).parent.name!=phases[-1]['identity']:raise ValueError('Unexpected checkpoint parent')
                elif '--resume' in args:raise ValueError('Root model unexpectedly resumes another fit')
                initial.add(report['initial_model_digest']);corpus.add(report['signature']['corpus_sha256'])
                if report['parameters']!=150436608:raise ValueError('Parameter mismatch')
                actual=report['updates']*report['flops_per_global_step']['training_matmul_flops']
                probe=report['flops_per_global_step']['training_matmul_flops']//report['accumulation']
                targets+=report['phase_real_targets'];updates+=report['updates'];flops+=actual;probes+=probe
                wall+=closure['wall_seconds'];warm+=report['warm_seconds'];assigned+=closure['assigned_chip_hours']
                phases.append({'identity':name,'phase_targets':report['phase_real_targets'],
                               'phase_updates':report['updates'],'launch_wall_seconds':closure['wall_seconds']})
                reports.append(report);parent=report;phase_union.add(name)
            last=reports[-1]
            if len(initial)!=1 or len(corpus)!=1 or targets!=last['total_real_targets'] or flops!=last['total_training_matmul_flops']:
                raise ValueError('Retained source/work history does not reproduce endpoint counters')
            if updates!=int(names[-1].rsplit('u',1)[1]):raise ValueError('Optimizer-update history differs')
            canonical=1 if arm=='ordinary' else 2
            score_path=a.root/'runs'/f'{names[-1]}-dev-r{canonical}'/'report.json'
            score=json.loads(score_path.read_text()) if score_path.exists() else None
            rows.append({'arm':arm,'seed':seed,'endpoint':endpoint,'unique_parameters':last['parameters'],
                'real_targets':targets,'optimizer_updates':updates,'training_update_matmul_flops':flops,
                'known_native_probe_matmul_flops':probes,'retained_training_launch_seconds':wall,
                'retained_training_warm_seconds':warm,'retained_assigned_chip_hours':assigned,
                'real_targets_per_training_launch_second':targets/wall,
                'dev_nll':score['metrics']['weighted_nll'] if score else None,
                'native_inference_forward_matmul_flops_per_window':score['forward_flops_per_window'] if score else None,
                'phases':phases})
    lookup={(r['arm'],r['seed'],r['endpoint']):r for r in rows};comparisons=[]
    for seed in [103,211,307]:
        candidate=lookup.get(('mixture',seed,'1B'));control=lookup.get(('ordinary',seed,'matched'))
        if not candidate or not control or candidate['dev_nll'] is None or control['dev_nll'] is None:continue
        shortfall=1-control['training_update_matmul_flops']/candidate['training_update_matmul_flops']
        if not 0<=shortfall<.00014:raise ValueError('Incorrect rounded update-work control')
        comparisons.append({'seed':seed,'ordinary_update_work_shortfall_fraction':shortfall,
              'mixture_retained_launch_wall_ratio':candidate['retained_training_launch_seconds']/control['retained_training_launch_seconds'],
              'mixture_inference_matmul_ratio':candidate['native_inference_forward_matmul_flops_per_window']/control['native_inference_forward_matmul_flops_per_window'],
              'mixture_dev_gain_nats':control['dev_nll']-candidate['dev_nll']})
    union_cost=sum(json.loads((a.root/'snapshots'/n/'closure.json').read_text())['assigned_chip_hours'] for n in phase_union)
    result={'status':'COMPLETE_RETAINED_PATH_RESOURCES' if not pending else 'PARTIAL_RETAINED_PATH_RESOURCES',
       'updated_unix':time.time(),'models':rows,'pending':pending,'matched_work_comparisons':comparisons,
       'unique_retained_phase_union_assigned_chip_hours':union_cost,
       'scope':'Per-model retained training paths only, including launch/CPU-probe/checkpoint wall overhead but excluding separate DEV/TEST scoring. The ordinary1B decay branch is not an ancestor of its1.7B control. Shared prefixes appear in both model rows; DO NOT sum those rows as campaign cost. The phase union deduplicates shared ancestors but still excludes unused/stopped attempts, tuning, readouts and CPU-only tails, all recorded in the separate execution ledger. Update matmuls and known native-probe work are separate; elementwise/optimizer/communication/compiler/CPU math is not in those FLOP counters. Inference figures are estimated executed forward matmuls, not generation latency.'}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Retained model paths and resource controls','',result['scope'],'',
       '| Arm | Seed | Endpoint | Real targets | Estimated update matmuls | Retained launch minutes | DEV NLL |',
       '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for r in rows:
        score='pending' if r['dev_nll'] is None else f"{r['dev_nll']:.6f}"
        lines.append(f"| {r['arm']} | {r['seed']} | {r['endpoint']} | {r['real_targets']:,} | {r['training_update_matmul_flops']:.5e} | {r['retained_training_launch_seconds']/60:.2f} | {score} |")
    lines+=['',f'{len(rows)} completed training paths; {len(pending)} still pending.',
            '','[Machine-readable paths](summary.json) · [Whole-campaign execution ledger](../execution-costs.json).']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'completed_paths':len(rows),'pending':len(pending),'matched':comparisons}),flush=True)


if __name__=='__main__':main()
