"""Bounded exploratory trained-one-pass control on the unchanged149 targets."""
import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA','0')
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

from compute_trainium.code_context_control import compare_trained_control
from compute_trainium.code_context_views import views
from compute_trainium.parcae_candidate import ParcaeCandidateConfig,matmul_ledger
from compute_trainium.parcae_training_candidate import initial_state
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.audit_parcae_precise_training_point import make_reference
from scripts.qualify_parcae_candidate_cpu import PIN
from scripts.qualify_native import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();started=time.time()
    parent_file=a.root/'parcae-low-depth-pilot-v1/main-plan-recovery-v2.json';parent=json.loads(parent_file.read_text())
    selection=a.root/'natural-code-context-v1';panel=selection/'scoring-panel-v3.json'
    selected=json.loads((selection/'scoring-views-v3-report.json').read_text())
    qualified=json.loads((selection/'intervention-qualification-v3.json').read_text())
    if (selected['status']!='FROZEN_PREFIX_ONLY_NATURAL_CODE_CONTEXT_VIEWS'
            or qualified['status']!='PASS_POSITION_NOISE_PRESERVING_CONTEXT_INTERVENTION'
            or file_sha256(panel)!=selected['panel_sha256'] or qualified['panel_sha256']!=selected['panel_sha256']):
        raise ValueError('Original qualified authentic-code views required')
    gates=json.loads(Path('results/parcae_low_depth_pilot_20261024/main-recovery-v2/gradient-progress.json').read_text())['records']
    points={}
    for readout,phase in [('matched','finish'),('equal','equal')]:
        gate=next(r for r in gates if r['arm']=='fixed1' and r['readout']==readout)
        case_id='fixed1-'+('main' if readout=='matched' else 'equal')+'-s1231';case=parent['cases'][case_id]
        checkpoint=a.root/f'runs/low-depth-main-r2-fixed1-{phase}/checkpoint-u{case["total_updates"]}.pt'
        candidate_dir=selection/'evaluation-execution-recovery-v3'/readout
        candidate=json.loads((candidate_dir/'report.json').read_text())
        if (gate['independent_cpu']['status']!='PASS_INDEPENDENT_MATURE_LOW_DEPTH_GRADIENT'
                or file_sha256(checkpoint)!=gate['independent_cpu']['checkpoint_sha256']
                or candidate['status']!='COMPLETE_FROZEN_NATURAL_CODE_CONTEXT_READOUT'
                or candidate['panel_sha256']!=selected['panel_sha256'] or candidate['main_plan_sha256']!=file_sha256(parent_file)):
            raise ValueError('Exact independently qualified control and same-panel candidate required')
        points[readout]={'case_id':case_id,'checkpoint':str(checkpoint.relative_to(a.root)),
            'checkpoint_sha256':gate['independent_cpu']['checkpoint_sha256'],'gradient_receipt':gate,
            'candidate_directory':str(candidate_dir.relative_to(a.root)),
            'candidate_report_sha256':file_sha256(candidate_dir/'report.json'),
            'candidate_private_rows_sha256':file_sha256(candidate_dir/'paired-private.json')}
    out=a.root/'natural-code-onepass-control-v1';public=Path('results/natural_code_context_20261024/onepass-control-v1')
    sources={**parent['source_sha256']}
    for f in ['scripts/evaluate_code_context_onepass_control.py','compute_trainium/code_context_control.py',
              'compute_trainium/code_context_views.py','tests/test_code_context_control.py']:
        sources[f]=file_sha256(f)
    plan={'identity':'natural-code-onepass-control-v1','registered_unix':started,
        'parent_plan_sha256':file_sha256(parent_file),'panel_sha256':file_sha256(panel),'points':points,
        'source_sha256':sources,'seconds_limit':1200,'prior_candidate_outcomes_seen':True,
        'supported_control_depth':1,'aggregation':'Original one-third distance bands,paired repository bootstrap5000 draws/seed1381',
        'assigned_chips':0,'optimizer_updates':0,'test_accessed':False,'terminal_alpha_spent':0,
        'scope':'Exploratory trained-one-pass control added after the original candidate mechanism readouts. Same149 authentic identifiers,full/short positions and prefix-only noise,no target/source/weight selection changes. Both original matched-matrix and equal-data checkpoints;one training seed and consumedDEV only. Does not replace the original primary result or establish general-text/terminal success.'}
    out.mkdir(exist_ok=False);public.mkdir(parents=True,exist_ok=False);write_state(out/'plan.json',plan);write_state(public/'plan.json',plan)
    result={'status':'STOPPED_TRAINED_ONEPASS_CONTEXT_CONTROL'};records=[];matrix=0;calls=0
    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if (time.time()-started>plan['seconds_limit'] or available<544*1024**3
                or min(shutil.disk_usage(out).free,shutil.disk_usage('/').free)<35*1024**3):raise RuntimeError('CPU code-control reserve/cap')
    try:
        guard();torch.set_num_threads(8);torch.set_num_interop_threads(1)
        upstream=a.root/'upstream-parcae-69284c1'
        if (subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=PIN
                or subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip()):raise ValueError('Clean pinned upstream required')
        sys.path.insert(0,str(upstream));config=ParcaeCandidateConfig(**parent['model'])
        corpus=a.root/'variable-depth-fresh-v1/dev.json';prep=json.loads((selection/'plan.json').read_text())
        if file_sha256(corpus)!=prep['source_sha256']:raise AssertionError('Original authentic-code sources changed')
        docs={d['id']:d for d in json.loads(corpus.read_text())};identifiers=json.loads(panel.read_text())['records']
        if len(identifiers)!=149:raise AssertionError('Keep the original identifier selection')
        for readout,point in points.items():
            guard();begin=time.time();case=parent['cases'][point['case_id']]
            saved=torch.load(a.root/point['checkpoint'],map_location='cpu',mmap=True,weights_only=False)
            if (saved['qualification_only'] is not False or saved['case_id']!=point['case_id']
                    or saved['signature']['plan_sha256']!=plan['parent_plan_sha256'] or saved['update']!=case['total_updates']):
                raise ValueError('Exact complete one-pass endpoint required')
            state_digest=digest({'model':dict(sorted(saved['model'].items())),'optimizer':saved['optimizer']})
            if state_digest!=saved['model_optimizer_digest']:raise AssertionError('Complete control checkpoint readback differs')
            model=make_reference(config,True).eval();model.load_state_dict(saved['model']);del saved
            before=digest(dict(sorted(model.state_dict().items())));current=None;model.initialize_state=lambda _:current;rows=[]
            with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
                for i,identifier in enumerate(identifiers):
                    guard();d=docs[identifier['document_id']]
                    if hashlib.sha256(json.dumps(d['tokens']).encode()).hexdigest()!=identifier['original_document_token_sha256']:
                        raise AssertionError('Authentic source tokens changed')
                    noise=initial_state(config,identifier['state_key'],case['state_seed']);losses={}
                    for label,view in views(d['tokens'],identifier).items():
                        tokens=torch.tensor([view['inputs']]);targets=torch.tensor(view['targets']);positions=torch.tensor(view['positions'])
                        current=noise[:,view['state_start']:view['state_start']+len(view['inputs'])].clone()
                        logits=model(tokens,position_ids=positions,num_steps_pair=torch.tensor([0,1]),return_logits=True)['logits'][0,view['prediction_positions']].double()
                        value=(logits.logsumexp(-1)-logits.gather(-1,targets[:,None]).squeeze(-1)).sum()
                        if not torch.isfinite(value):raise FloatingPointError('Nonfinite one-pass identifier score')
                        losses[label]=float(value);calls+=1
                        matrix+=matmul_ledger(config,len(view['inputs']),total_steps=1,backward_cap=1)['forward_matmul_flops']
                    rows.append({'id':identifier['id'],'group':identifier['group'],'band':identifier['band'],
                        'real_targets':identifier['target_end_token']-identifier['target_start_token'],'nll_sum':losses})
                    if (i+1)%20==0:write_state(public/'progress.json',{'readout':readout,'identifiers':i+1,'wall_seconds':time.time()-started})
            if digest(dict(sorted(model.state_dict().items())))!=before:raise AssertionError('Control scoring changed weights')
            candidate_dir=a.root/point['candidate_directory']
            if (file_sha256(candidate_dir/'report.json')!=point['candidate_report_sha256']
                    or file_sha256(candidate_dir/'paired-private.json')!=point['candidate_private_rows_sha256']):
                raise AssertionError('Previously fitted candidate scores changed')
            candidate=json.loads((candidate_dir/'paired-private.json').read_text())['records']
            record={'readout':readout,'comparison':compare_trained_control(rows,candidate),
                'control_checkpoint_sha256':point['checkpoint_sha256'],'control_depth':1,
                'control_parameters':case['parameters'],'complete_state_readback_exact':True,'model_unchanged':True,
                'wall_seconds':time.time()-begin}
            records.append(record);write_state(public/(readout+'.json'),record)
            write_state(out/(readout+'-private.json'),{'records':rows});del model;gc.collect()
        if any(file_sha256(f)!=sha for f,sha in sources.items()):raise AssertionError('Frozen code-control source changed')
        result={'status':'COMPLETE_TRAINED_ONEPASS_CONTEXT_CONTROL','records':records}
    except BaseException as error:result['error']=str(error);raise
    finally:
        result.update(records=records,wall_seconds=time.time()-started,forward_calls=calls,estimated_forward_matrix_ops=matrix,
            assigned_chips=0,optimizer_updates=0,test_accessed=False,terminal_success=False,scope=plan['scope'])
        write_state(out/'report.json',result);write_state(public/'report.json',result)
        print(json.dumps({'status':result['status'],'wall_seconds':result['wall_seconds']}),flush=True)


if __name__=='__main__':main()
