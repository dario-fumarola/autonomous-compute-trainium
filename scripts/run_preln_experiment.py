"""New recurrent fits against bit-equivalent, preserved ordinary controls."""
import argparse
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from compute_trainium.preln_recurrence import PreNormRecurrenceConfig
from compute_trainium.training import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--identity',default='preln-recurrent-v1');p.add_argument('--deadline-unix',type=float,required=True)
    args=p.parse_args();root=Path(os.environ['RESEARCH_ROOT']);container_root=os.environ['RESEARCH_CONTAINER_ROOT']
    original=root.parent
    qualified=json.loads((root/'preln-qualification.json').read_text())
    if qualified['status']!='PASS_PRELN_NATIVE_QUALIFICATION':raise ValueError('Native qualification incomplete')
    cpu=json.loads((original/'recurrence/runs/preln-exact-cpu-001/report.json').read_text())
    if not cpu['ordinary_logits_bit_identical'] or not cpu['ordinary_full_gradients_bit_identical']:
        raise ValueError('Ordinary control equivalence failed')
    reference=json.loads((args.reference/'progress.json').read_text())
    if not (args.reference/'complete.json').exists():raise ValueError('Ordinary reference incomplete')
    ref_spec=reference['specification']
    manifest=json.loads((args.data/'manifest.json').read_text())
    if (manifest['status']!='PASS_FROZEN_NATURAL_CORPUS' or manifest['source_group_overlap']
        or file_sha256(args.data/'manifest.json')!=ref_spec['data_manifest_sha256']):
        raise ValueError('Reference and new training data differ')
    ordinary=[dict(r) for r in reference['records'] if r['arm']=='ordinary']
    if len(ordinary)!=9:raise ValueError('All three original seeds/milestones required')
    reference_lrs=json.loads((args.reference/'selected-lrs.json').read_text())['selected_lrs']
    if reference_lrs['ordinary']!=.001:raise ValueError('Frozen ordinary LR changed')
    args.output.mkdir(exist_ok=False);started=time.time()
    prefix=np.load(args.data/'packed/prefix_counts.npy',mmap_mode='r')
    def targets(step):return int(prefix[step*256].sum())
    files=['compute_trainium/model.py','compute_trainium/titan.py','compute_trainium/recurrence.py',
           'compute_trainium/preln_recurrence.py','compute_trainium/continuation.py','compute_trainium/training.py',
           'compute_trainium/data.py','compute_trainium/metrics.py','scripts/train_distributed.py','scripts/evaluate_checkpoint.py']
    rate=qualified['warm_real_targets_per_second']
    estimated=(3*(targets(3840)+targets(2269)-targets(2042))+3*targets(384))/rate*1.3+21*150
    if time.time()+estimated>args.deadline_unix:raise ValueError('Measured primary budget exceeds deadline')
    spec={'identity':args.identity,'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
          'protocol_sha256':file_sha256('docs/PRELN_RECURRENT_PROTOCOL.md'),
          'code_sha256':{f:file_sha256(f) for f in files},'seeds':[23,47,71],'tuning_seed':11,
          'lrs':[.0003,.001,.003],'ordinary_selected_lr':.001,
          'data_manifest_sha256':file_sha256(args.data/'manifest.json'),
          'dev_sha256':file_sha256(args.data/'dev.json'),'sealed_test_sha256':file_sha256(args.data/'test-sealed.json'),
          'reference_specification_sha256':file_sha256(args.reference/'specification.json'),
          'reused_reference_runs':ordinary,'primary_token_endpoint_update':3840,
          'compute_endpoint_update':2269,'compute_fork_update':2042,
          'boundaries_real_targets':{str(s):targets(s) for s in [16,40,346,384,1152,2042,2269,3456,3840]},
          'parameters':PreNormRecurrenceConfig().parameter_count(),'architecture':'pre-norm shared core, fixed two passes',
          'world':64,'local_batch':2,'accumulation':2,'data_seed':23,
          'flops_per_global_update':{str(r):PreNormRecurrenceConfig(repetitions=r).flops(1024,1024,256) for r in [1,2]},
          'budget':{'qualified_targets_per_second':rate,'conservative_seconds':estimated,'deadline_unix':args.deadline_unix},
          'claim_threshold_nats':.01,'longer_diagnostic':{'seed':23,'endpoint_update':11520,'requires':'Qualified append-only natural data and measured remaining time; selection independent of primary gain'},
          'test_sealed':True,'reused_controls_disclosed':True}
    (args.output/'specification.json').write_text(json.dumps(spec,indent=2)+'\n')
    common=['--size','small','--architecture','preln','--repetitions','2','--context','1024','--batch','2',
            '--accumulation','2','--precision','bf16','--data-seed','23','--reference-threads','8',
            '--corpus','/work/natural-v2-1050m/source.json','--packed-corpus','/work/natural-v2-1050m/packed',
            '--dev-corpus','/work/natural-v2-1050m/dev.json','--warmup','4']
    records=[]
    def persist():
        (args.output/'progress.json').write_text(json.dumps({'specification':spec,'records':records,
            'elapsed_seconds':time.time()-started},indent=2)+'\n')
    def run(name,arguments,steps,cohort,resume=None,decay=None,scientific=False):
        if time.time()>args.deadline_unix:raise TimeoutError('Campaign deadline')
        for f,sha in spec['code_sha256'].items():
            if file_sha256(f)!=sha:raise ValueError('Frozen scientific source changed: '+f)
        plan={'qualification_only':not scientific,'world':64,'training_memory_gib':288,'evaluation_memory_gib':128,
              'timeout_seconds':3600,'evaluation_cohort':cohort,'training_arguments':arguments,
              'phases':[{'run':name,'steps':steps,'pauses':0}]}
        if scientific:plan['scientific_manifest']=str(args.output/'launch-gates.json')
        if resume:plan['resume']=resume
        if decay:plan['phases'][0].update(decay_start=targets(decay[0]),decay_end=targets(decay[1]),branch=True)
        path=args.output/(name+'-plan.json');path.write_text(json.dumps(plan,indent=2)+'\n')
        subprocess.run(['python3','-m','scripts.run_phases','--plan',str(path),'--output',str(args.output/(name+'-controller'))],check=True)
        train=json.loads((root/'runs'/name/'report.json').read_text());dev=json.loads((root/'runs'/(name+'-dev')/'report.json').read_text())
        return train,dev,f'{container_root}/runs/{name}/checkpoint.pt'
    try:
        trials=[]
        for i,lr in enumerate(spec['lrs']):
            name=f'{args.identity}-lr-{i}'
            a=[*common,'--purpose','tuning','--seed','11','--lr',str(lr),'--lr-warmup-tokens',str(targets(16)),
               '--decay-start',str(targets(346)),'--decay-end',str(targets(384))]
            train,dev,_=run(name,a,384,'tune')
            trials.append({'run':name,'lr':lr,'weighted_nll':dev['metrics']['weighted_nll'],'initial_model_digest':train['initial_model_digest']})
            (args.output/'tuning.json').write_text(json.dumps(trials,indent=2)+'\n')
        selected=min(trials,key=lambda d:d['weighted_nll'])
        if selected['weighted_nll']>=5.0:
            (args.output/'learning-gate-failed.json').write_text(json.dumps(selected,indent=2)+'\n')
            raise ValueError('Predeclared 100M-target learning gate failed')
        lr=selected['lr']
        (args.output/'selected-lr.json').write_text(json.dumps({'selected_lr':lr,'frozen_unix':time.time(),
            'criterion':'Minimum tuning DEV NLL; below predeclared learning gate 5.0'},indent=2)+'\n')
        gates={k:True for k in ['immutable_sources','near_duplicate_audit','frozen_order_sha256','lr_selected_at_this_batch','measured_budget','trainer_qualified']}
        gates.update(learning_gate_passed=True,specification_sha256=file_sha256(args.output/'specification.json'))
        (args.output/'launch-gates.json').write_text(json.dumps(gates,indent=2)+'\n');persist()
        previous={s:None for s in spec['seeds']};old_step={s:0 for s in spec['seeds']}
        for step,role in [(384,'trunk'),(1152,'trunk'),(2042,'compute_trunk'),(2269,'compute_endpoint'),(3456,'token_trunk'),(3840,'token_endpoint')]:
            for seed in spec['seeds']:
                name=f'{args.identity}-s{seed}-u{step}'
                a=[*common,'--purpose','scientific','--seed',str(seed),'--lr',str(lr),
                   '--lr-warmup-tokens',str(targets(40)),'--checkpoint-every','384']
                decay=(2042,2269) if role=='compute_endpoint' else ((3456,3840) if role=='token_endpoint' else None)
                train,dev,path=run(name,a,step-old_step[seed],'curve',previous[seed],decay,True)
                initial=next(r['initial_model_digest'] for r in ordinary if r['seed']==seed)
                if train['initial_model_digest']!=initial:raise AssertionError('Paired initial weights differ')
                if train['total_real_targets']!=targets(step):raise AssertionError('Matched real target exposure differs')
                row={'run':name,'arm':'recurrent','seed':seed,'update':step,'role':role,
                     'real_tokens':train['total_real_targets'],'metrics':dev['metrics'],'checkpoint':path,
                     'initial_model_digest':initial,'training_matmul_flops':train['total_training_matmul_flops'],
                     'completed_unix':time.time()}
                records.append(row)
                if role!='compute_endpoint':previous[seed]=path;old_step[seed]=step
                persist();print(json.dumps({'event':'curve','seed':seed,'update':step,'role':role,'weighted_nll':dev['metrics']['weighted_nll']}),flush=True)
        (args.output/'complete.json').write_text(json.dumps({'status':'COMPLETE_PRELN_RECURRENT_PRIMARY',
             'records':len(records),'elapsed_seconds':time.time()-started,'ended_unix':time.time()},indent=2)+'\n')
    except BaseException as error:
        (args.output/'failure.json').write_text(json.dumps({'error':type(error).__name__,'message':str(error),
            'records':len(records),'elapsed_seconds':time.time()-started},indent=2)+'\n');raise


if __name__=='__main__':main()
