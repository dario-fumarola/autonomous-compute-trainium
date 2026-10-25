"""Admit bounded discovery stages only after independent measured hardware gates."""
import argparse
import json
import time
from pathlib import Path

from compute_trainium.future_span_scientific_state import SCIENCE
from compute_trainium.future_span_study_plan import CELLS,RATES,measured_budget
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def prerequisites(root):
    root=Path(root);files={};receipts={}
    for folder,status in [('future-span-qualification-v1','PASS_COMPLETE_FUTURE_SPAN_TRAINING_QUALIFICATION'),
                          ('future-span-driver-v1','PASS_COMPLETE_FUTURE_SPAN_SCIENTIFIC_DRIVER')]:
        plan_file=root/folder/'plan.json';report_file=root/folder/'report.json'
        report=json.loads(report_file.read_text());plan=json.loads(plan_file.read_text())
        if report['status']!=status or report['plan_sha256']!=file_sha256(plan_file):
            raise ValueError('Complete same-plan hardware and scientific-driver passes required')
        for path,sha in plan['source_sha256'].items():
            if file_sha256(path)!=sha or (path in files and files[path]!=sha):
                raise AssertionError('Qualified frozen executable source changed')
            files[path]=sha
        receipts[folder]={'plan_sha256':file_sha256(plan_file),'report_sha256':file_sha256(report_file)}
    prep=root/'future-span-study-preparation-v1';manifest=json.loads((prep/'manifest.json').read_text())
    report=json.loads((prep/'report.json').read_text())
    if (report['status']!='COMPLETE_FRESH_FUTURE_SPAN_SCIENTIFIC_PREPARATION'
            or report['manifest_sha256']!=file_sha256(prep/'manifest.json')):
        raise ValueError('Exact fresh independent initializers/source ledgers required')
    for path,sha in manifest['source_sha256'].items():
        if file_sha256(path)!=sha or (path in files and files[path]!=sha):raise AssertionError('Fresh preparation source changed')
        files[path]=sha
    if {r['seed'] for r in manifest['initial_states'].values()}!={4127,5231,7283}:
        raise ValueError('The three prospective initializer seeds must remain fixed')
    for info in manifest['initial_states'].values():
        if file_sha256(prep/info['file'])!=info['sha256']:raise AssertionError('Fresh initializer changed')
    for table in manifest['tables'].values():
        for key in ['depth_table','work_table']:
            if file_sha256(prep/table[key])!=table[key+'_sha256']:raise AssertionError('Fresh source/depth/work ledger changed')
    auxiliary=manifest['auxiliary_data']
    if file_sha256(root/auxiliary['directory']/auxiliary['prefix_file'])!=auxiliary['prefix_sha256']:
        raise AssertionError('Fresh auxiliary-prefix counts changed')
    receipts[prep.name]={'manifest_sha256':file_sha256(prep/'manifest.json'),'report_sha256':file_sha256(prep/'report.json')}
    return files,receipts,manifest


def register(root,stage):
    root=Path(root);sources,receipts,manifest=prerequisites(root)
    hardware=json.loads((root/'future-span-qualification-v1/report.json').read_text())
    budget=measured_budget(hardware,stage=stage)
    if not budget['fits_registered_cap']:raise ValueError('Measured conservative stage budget exceeds its prospective cap')
    chosen=None
    if stage=='main':
        tune_file=root/'future-span-tune-v1/report.json';tune=json.loads(tune_file.read_text())
        tune_plan_file=root/'future-span-tune-v1/plan.json'
        if (tune['status']!='COMPLETE_BOUNDED_FUTURE_SPAN_TUNING' or tune['plan_sha256']!=file_sha256(tune_plan_file)
                or set(tune['selected'])!=set(CELLS) or len(tune['selected_gradient_gates'])!=4
                or any(r['independent_cpu']['status']!='PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT'
                       for r in tune['selected_gradient_gates'])):
            raise ValueError('All frozen TUNE opportunities and four selected trained-gradient passes required')
        chosen=tune['selected'];receipts['future-span-tune-v1']={'plan_sha256':file_sha256(tune_plan_file),'report_sha256':file_sha256(tune_file)}
    for path in ['compute_trainium/future_span_evaluation.py','compute_trainium/future_span_gradient_point.py',
                 'compute_trainium/future_span_study_plan.py','compute_trainium/future_span_analysis.py',
                 'scripts/register_future_span_study.py',
                 'scripts/run_future_span_study.py','scripts/evaluate_future_span_cpu.py',
                 'scripts/capture_future_span_gradient.py','scripts/audit_future_span_gradient.py',
                 'tests/test_future_span_evaluation.py','tests/test_future_span_gradient_point.py',
                 'tests/test_future_span_study_plan.py','tests/test_future_span_study_admission.py',
                 'tests/test_future_span_analysis.py']:
        sha=file_sha256(path)
        if path in sources and sources[path]!=sha:raise AssertionError('Registered study source conflicts with qualification')
        sources[path]=sha
    identity='future-span-'+stage+'-v1';out=root/identity;public=Path('results/future_span_'+stage+'_20261025')
    # All gates precede creation. No empty directory can be mistaken for a
    # scientific registration after a missing or failed prerequisite.
    if out.exists() or public.exists():raise FileExistsError('Preserve an existing scientific identity')
    prep=root/'future-span-study-preparation-v1'
    initials={key:{**info,'file':'../'+prep.name+'/'+info['file']} for key,info in manifest['initial_states'].items()}
    cases={};identities=[];seeds=[4127] if stage=='tune' else [5231,7283]
    milestone='tuning' if stage=='tune' else 'main'
    for seed in seeds:
        for cell in CELLS:
            arm='recurrent' if cell.startswith('recurrent') else 'one';table=manifest['tables'][arm];m=table['milestones'][milestone]
            for lr in RATES if stage=='tune' else [chosen[cell]['peak_lr']]:
                key=f'{cell}-s{seed}-lr{round(lr*1e6)}'
                case={'cell':cell,'arm':arm,'objective_weight':.1 if cell.endswith('future') else 0.,
                    'seed':seed,'state_seed':seed+2009,'initial_state':f's{seed}-family',
                    'training_parameters':initials[f's{seed}-family']['training_parameters'],
                    'peak_lr':lr,'warmup_targets':50_000_000,'cooldown_start_update':m['cooldown_start_update'],
                    'cooldown_start_targets':m['cooldown_start_targets'],'total_updates':m['updates'],'end_targets':m['real_targets'],
                    'end_anchors':m['valid_anchors'],'training_dense_matrix_ops':m['estimated_dense_matrix_ops'],
                    'depth_table':'../'+prep.name+'/'+table['depth_table'],'depth_table_sha256':table['depth_table_sha256'],
                    'work_table':'../'+prep.name+'/'+table['work_table'],'work_table_sha256':table['work_table_sha256'],
                    'checkpoint_updates':[288,576] if stage=='tune' else [576,960,1920],
                    'segment_ends':[576] if stage=='tune' else [960,1920],
                    'gradient_updates':[576] if stage=='tune' else [1920],
                    'primary_inference_depth':1 if arm=='one' else 2}
                cases[key]=case
                identities.extend([identity+'-'+key+'-'+phase for phase in (['fit','gradient'] if stage=='tune' else ['first','finish','gradient'])])
    source= root/'variable-depth-fresh-v1/dev.json'
    plan={'identity':identity,'registered_unix':time.time(),'purpose':SCIENCE,'stage':stage,
        'launch_gates_complete':True,'launch_receipts':receipts,'source_sha256':sources,
        'worker_source_sha256':{p:sha for p,sha in sources.items() if p.startswith(('compute_trainium/','scripts/','configs/','third_party/'))},
        'model':manifest['model'],'future_config':manifest['future_config'],'gradient_rule':'all_sampled_steps',
        'precision':'fp16_body_fp32_heads_masters','accumulation':16,'routing_seed':manifest['routing_seed'],
        'data':manifest['data'],'auxiliary_data':manifest['auxiliary_data'],'initial_states':initials,
        'evaluation':{'scope':'consumed_development_only','corpus':'variable-depth-fresh-v1/dev.json',
            'corpus_sha256':file_sha256(source),'weights':{'web':.6,'math':.2,'code':.2},
            'noise_policy':'original_prefix_window_index_only_fixed1024_draw'},
        'cases':cases,'selected_rates':chosen,'learning_rate_candidates':list(RATES),
        'root_comm_ports':reserve_ports(root,identities),'seconds_limit':budget['execution_seconds_cap'],
        'phase_seconds_limit':2400 if stage=='tune' else 3600,'cpu_phase_seconds_limit':1200,
        'measured_budget':budget,'minimum_useful_gain_nats':.01,
        'primary_interaction':'(L(one-future)-L(recurrent-future))-(L(one-ntp)-L(recurrent-ntp))',
        'absolute_candidate_gates':['recurrent-future improves one-future >=0.01','recurrent-future improves recurrent-ntp >=0.01',
            'recurrent-future supported depth1-to2 improves >=0.01'],
        'primary_interval_scope':'Two pilot seeds only;source-group uncertainty reported separately;no terminal significance or confirmation.',
        'strongest_baseline_required':'Before practical or terminal claim,qualify ordinary/cheap-connection with the same objective and independently annealed matched-compute endpoints.',
        'test_accessed':False,'terminal_alpha_spent':0,
        'scope':'Bounded equal-data four-cell objective discovery. Fresh random weights,common source order,all sampled steps differentiated,training-only head removed at scoring. Complete cooldown at every primary endpoint. No fresh TEST,confirmation,router,teacher,synthetic corpus or compute-efficiency claim.'}
    out.mkdir();public.mkdir(parents=True);write_state(out/'plan.json',plan);write_state(public/'plan.json',plan)
    return out,public,plan


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['tune','main'],required=True)
    a=p.parse_args();out,_,plan=register(a.root,a.stage)
    print(json.dumps({'identity':plan['identity'],'plan_sha256':file_sha256(out/'plan.json'),'cases':len(plan['cases']),
        'predicted_seconds_with_margin':plan['measured_budget']['predicted_seconds_with_margin'],'launched':False}))


if __name__=='__main__':main()
