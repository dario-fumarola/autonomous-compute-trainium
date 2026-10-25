"""Scientific-driver integration: reuse exact initial gold and check real recovery."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA','0')
import numpy as np
import torch

from compute_trainium.future_span_reference_optimizer import reference_optimizer,fp32_reference_execution
from compute_trainium.future_span_scientific_state import INTEGRATION,signature,validate_checkpoint
from compute_trainium.parcae_recovery_check import check_replay,moments
from compute_trainium.parcae_scientific_state import configure_optimizer
from compute_trainium.parcae_update_qualification import named_moments,compare_update,compare_moments
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.audit_future_span_qualification import construct_reference,gradient_check
from scripts.qualify_native import digest
from scripts.qualify_parcae_candidate_cpu import PIN
from scripts.run_parcae_optimizer_native import step_comparison,step_passes


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--cell',required=True);parser.add_argument('--captures',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--upstream',type=Path,required=True)
    args=parser.parse_args();started=time.time();args.output.mkdir(parents=True,exist_ok=False)
    result={'status':'STOPPED_FUTURE_SPAN_DRIVER_AUDIT'};steps=0
    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if (available<544*1024**3 or min(shutil.disk_usage(args.output).free,shutil.disk_usage('/').free)<35*1024**3
                or time.time()-started>600):raise RuntimeError('Independent driver audit reserve/cap')
    try:
        guard();torch.set_num_threads(12)
        if file_sha256(args.plan)!=args.plan_sha:raise AssertionError('Frozen driver plan changed')
        plan=json.loads(args.plan.read_text())
        if plan['purpose']!=INTEGRATION or any(file_sha256(p)!=sha for p,sha in plan['source_sha256'].items()):
            raise ValueError('Unchanged finite driver qualification required')
        if (subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip()!=PIN
                or subprocess.check_output(['git','-C',str(args.upstream),'status','--porcelain'],text=True).strip()):
            raise ValueError('Clean pinned independent reference required')
        sys.path.insert(0,str(args.upstream));captures=json.loads(args.captures.read_text())
        reports={};directories={}
        for phase,row in captures.items():
            directory=args.root/'runs'/row['identity'];file=directory/'report.json'
            if file_sha256(file)!=row['report_sha256']:raise AssertionError('Captured driver phase changed')
            report=json.loads(file.read_text())
            if (report['status']!='COMPLETE_REGISTERED_FUTURE_SPAN_SCIENTIFIC_SEGMENT' or report['plan_sha256']!=args.plan_sha
                    or report['purpose']!=INTEGRATION or not report['qualification_only'] or report['research_optimizer_updates']!=0
                    or report['cell']!=args.cell):raise ValueError('Driver phase identity differs')
            reports[phase]=report;directories[phase]=directory
        first=reports['first'];parent_case=plan['cases'][first['case_id']]
        initial_info=plan['initial_states'][parent_case['initial_state']];initial_file=args.plan.parent/initial_info['file']
        if file_sha256(initial_file)!=initial_info['sha256']:raise AssertionError('Qualified original initializer changed')
        initial=torch.load(initial_file,map_location='cpu',mmap=True,weights_only=False)
        hw_file=args.root/'future-span-qualification-v1/report.json'
        if file_sha256(hw_file)!=plan['hardware_qualification_sha256']:raise AssertionError('Hardware prerequisite changed')
        hardware=json.loads(hw_file.read_text());hw=next(r for r in hardware['records'] if r['case']['identity']==args.cell)
        hw_point=next(r for r in hw['native']['warm']['probes'] if r['update']==0)
        point=first['first_global_probe']
        if (hw['initial_cpu']['gold_fingerprint']['model_digest']!=initial_info['model_digest']
                or point['rows']!=hw_point['rows'] or point['source_cursor']!=hw_point['cursor']
                or point['real_targets']!=hw_point['real_targets'] or point['valid_anchors']!=hw_point['valid_anchors']):
            raise AssertionError('Driver first objective differs from the independently checked hardware state/source/noise/depth')
        gold_file=args.root/'future-span-qualification-v1/cpu'/args.cell/'warm-u0/cpu-global.pt'
        if file_sha256(gold_file)!=hw['initial_cpu']['cpu_global_sha256']:raise AssertionError('Independent initial gold changed')
        gold=torch.load(gold_file,map_location='cpu',mmap=True,weights_only=False)
        vector_file=directories['first']/point['gradient_file']
        if file_sha256(vector_file)!=point['gradient_sha256']:raise AssertionError('Driver global vector changed')
        actual=torch.load(vector_file,map_location='cpu',mmap=True,weights_only=False)
        gradient=gradient_check(gold['gradients'],actual['gradients'])
        loss_diff={k:actual[k]-gold[k] for k in ['ntp_nll','future_mean']}
        model=construct_reference(ParcaeCandidateConfig(**plan['model']),initial['model'])
        first_step_file=directories['first']/'after-first-update.pt'
        if file_sha256(first_step_file)!=first['first_update_sha256']:raise AssertionError('Actual first scientific-driver update changed')
        native_step=torch.load(first_step_file,map_location='cpu',mmap=True,weights_only=False)
        optimizer=reference_optimizer(model,native_step['optimizer']['recipe']['groups'])
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        for n,p in model.named_parameters():p.grad=actual['gradients'][n].clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False)
        configure_optimizer(optimizer,parent_case,point['real_targets'])
        with fp32_reference_execution():optimizer.step()
        steps+=1
        expected_step={'model':{n:p.detach() for n,p in model.named_parameters()},'moments':named_moments(model,optimizer,reference=True)}
        right={'model':{n:native_step['model'][n] for n in before},'moments':moments(native_step['optimizer'])}
        first_update=step_comparison(before,expected_step,before,right)
        head_names=[n for n in before if n.startswith('future_lm_head.')]
        first_update['future_head']={'update':compare_update(before,expected_step['model'],before,right['model'],names=head_names),
            'moments':compare_moments({n:expected_step['moments'][n] for n in head_names},{n:right['moments'][n] for n in head_names})}
        packing=args.root/plan['data']['packed'];packing_meta=json.loads((packing/'manifest.json').read_text())
        if (file_sha256(packing/'manifest.json')!=plan['data']['manifest_sha256']
                or file_sha256(packing/'prefix_counts.npy')!=packing_meta['files']['prefix_counts.npy']):
            raise AssertionError('Independent driver source-count manifest changed')
        counts=np.load(packing/'prefix_counts.npy',mmap_mode='r')
        aux=plan['auxiliary_data'];prefix=args.root/aux['directory']/aux['prefix_file']
        if file_sha256(prefix)!=aux['prefix_sha256']:raise AssertionError('Independent driver auxiliary counts changed')
        anchors=np.load(prefix,mmap_mode='r')
        states={}
        for phase,update in [('first',20),('finish',21),('replay',21),('branch',30)]:
            guard()
            report=reports[phase];row=next(r for r in report['checkpoint_records'] if r['update']==update)
            path=directories[phase]/f'checkpoint-u{update}.pt'
            if file_sha256(path)!=row['checkpoint_sha256']:raise AssertionError('Driver checkpoint changed')
            state=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
            if digest({'model':dict(sorted(state['model'].items())),'optimizer':state['optimizer']})!=row['full_state_digest']:
                raise AssertionError('Independent complete-state checkpoint readback differs')
            case=plan['cases'][report['case_id']];work_file=args.plan.parent/case['work_table']
            if file_sha256(work_file)!=case['work_table_sha256']:raise AssertionError('Driver counter table changed')
            domain=list(map(int,counts[update*512]-counts[0]));work=int(np.load(work_file,mmap_mode='r')[:update].sum())
            if phase in ['first','finish','replay']:
                validate_checkpoint(state,report['case_id'],case,signature(args.plan_sha,plan,case),
                    expected_targets=sum(domain),expected_domains=domain,expected_matrix_ops=work,expected_anchors=int(anchors[update*512]))
            else:
                if (state['real_targets']!=sum(domain) or state['valid_anchors']!=int(anchors[update*512])
                        or state['domain_targets']!=domain or state['cursor']!=update*512
                        or state['signature']!=signature(args.plan_sha,plan,case)
                        or state['estimated_dense_matrix_ops']!=work or state['stable_trunk']):
                    raise AssertionError('Completed cooldown-branch counters/stability differ')
            states[phase]=state
        parent_digest=next(r['full_state_digest'] for r in first['checkpoint_records'] if r['update']==20)
        if any(not reports[p]['complete_state_on_load_exact'] or reports[p]['recovered_state_digest']!=parent_digest
               or not reports[p]['cpu_rng_on_load_exact'] for p in ['finish','replay','branch']):
            raise AssertionError('Actual full-state/RNG loads failed')
        replay=check_replay(states['first'],states['finish'],states['replay'])
        if sum(r['actual_optimizer_calls_per_replica'] for r in reports.values())!=51:raise AssertionError('Driver repeated-update count differs')
        head_ok=all(v['relative_l2']<=.02 for v in first_update['future_head'].values())
        guard();passed=gradient['passes_two_percent_gate'] and all(abs(v)<=.003 for v in loss_diff.values()) and step_passes(first_update) and head_ok and replay['passes_two_percent_gate']
        result={'status':'PASS_FUTURE_SPAN_SCIENTIFIC_DRIVER_INTEGRATION' if passed else 'FAIL_FUTURE_SPAN_SCIENTIFIC_DRIVER_INTEGRATION',
            'cell':args.cell,'initial_full_gradient':gradient,'initial_loss_differences':loss_diff,
            'first_independent_optimizer_update':first_update,'replay':replay,
            'all_checkpoint_readbacks_exact':True,'all_actual_state_rng_loads_exact':True,
            'stable_prefix_branch_completed_own_decay':True,'actual_native_optimizer_calls_per_replica':51}
        if not passed:raise RuntimeError('Driver numerical/recovery gate failed')
    except BaseException as error:result['error']=str(error);raise
    finally:
        result.update(assigned_chips=0,optimizer_updates=0,prescribed_cpu_reference_optimizer_steps=steps,
            research_optimizer_updates=0,test_accessed=False,wall_seconds=time.time()-started,
            scope='Finite scientific-driver plumbing qualification. Exact initial neural gold reused only after matching512 source/depth/noise rows. Actual first optimizer update,all counters,full-state/RNG restore,21st-update numerical replay and independent shorter cooldown branch checked. No scientific quality result.')
        write_state(args.output/'report.json',result)


if __name__=='__main__':main()
