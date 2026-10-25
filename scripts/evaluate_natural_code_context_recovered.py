"""Same frozen four scores,with an explicit recovered gradient-receipt path.

Only execution plumbing changes;source selection,views,scoring and thresholds
remain the original preregistered implementation.
"""
import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('USE_TORCH_XLA','0')
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

from compute_trainium.code_context_views import views,summarize,group_intervals
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_training_candidate import initial_state
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts.audit_parcae_precise_training_point import make_reference
from scripts.qualify_native import digest
from scripts.qualify_parcae_candidate_cpu import PIN


def require_endpoint_gradient(path,readout,checkpoint_sha):
    rows=json.loads(path.read_text())['records']
    matched=[r for r in rows if r['arm']=='poisson2' and r['readout']==readout]
    if len(matched)!=1:raise ValueError('One exact endpoint gradient receipt required')
    receipt=matched[0]
    if (receipt['independent_cpu']['status']!='PASS_INDEPENDENT_MATURE_LOW_DEPTH_GRADIENT'
            or receipt['independent_cpu']['checkpoint_sha256']!=checkpoint_sha):
        raise ValueError('Exact endpoint independent gradient gate required')
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--checkpoint-sha',required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True)
    p.add_argument('--gradient-receipts',type=Path,required=True)
    p.add_argument('--readout',choices=['matched','equal'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();started=time.time()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(12);torch.set_num_interop_threads(1)
    selection_dir=a.root/'natural-code-context-v1';selection=json.loads((selection_dir/'scoring-views-v3-report.json').read_text())
    prep=json.loads((selection_dir/'plan.json').read_text());qualified=json.loads((selection_dir/'intervention-qualification-v3.json').read_text())
    if selection['status']!='FROZEN_PREFIX_ONLY_NATURAL_CODE_CONTEXT_VIEWS' or qualified['status']!='PASS_POSITION_NOISE_PRESERVING_CONTEXT_INTERVENTION':raise ValueError('Frozen adequate panel and intervention qualification required')
    if file_sha256(selection_dir/'scoring-panel-v3.json')!=selection['panel_sha256'] or qualified['panel_sha256']!=selection['panel_sha256']:raise AssertionError('Context selection panel changed')
    if file_sha256(a.plan)!=a.plan_sha or file_sha256(a.checkpoint)!=a.checkpoint_sha:raise AssertionError('Scientific checkpoint/plan changed')
    plan=json.loads(a.plan.read_text());case_id='poisson2-'+('main' if a.readout=='matched' else 'equal')+'-s1231';case=plan['cases'][case_id]
    if plan['purpose']!='natural_low_depth_scientific_pilot' or case['arm']!='poisson2':raise ValueError('Supported-exit candidate pilot only')
    receipt=require_endpoint_gradient(a.gradient_receipts,a.readout,a.checkpoint_sha)
    upstream=a.root/'upstream-parcae-69284c1'
    if subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()!=PIN or subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip():raise ValueError('Clean pinned upstream scoring required')
    sys.path.insert(0,str(upstream));saved=torch.load(a.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if (saved['qualification_only'] is not False or saved['signature']['plan_sha256']!=a.plan_sha or saved['case_id']!=case_id
        or saved['update']!=case['total_updates']):raise ValueError('Completed registered scientific endpoint required')
    state_digest=digest({'model':dict(sorted(saved['model'].items())),'optimizer':saved['optimizer']})
    if state_digest!=saved['model_optimizer_digest']:raise AssertionError('Full-state readback differs')
    c=ParcaeCandidateConfig(**plan['model']);model=make_reference(c,True).eval();model.load_state_dict(saved['model'],strict=True);del saved
    before=digest(dict(sorted(model.state_dict().items())));current=None;model.initialize_state=lambda _:current
    corpus=a.root/'variable-depth-fresh-v1/dev.json'
    if file_sha256(corpus)!=prep['source_sha256']:raise AssertionError('Frozen original source changed')
    docs={d['id']:d for d in json.loads(corpus.read_text())};points=json.loads((selection_dir/'scoring-panel-v3.json').read_text())['records'];rows=[]
    def guard():
        available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available<544*1024**3 or min(shutil.disk_usage(a.output).free,shutil.disk_usage('/').free)<35*1024**3:
            raise RuntimeError('Code-context CPU memory/disk reserve')
        if time.time()-started>1200:raise TimeoutError('Code-context endpoint20minute cap')
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        for i,point in enumerate(points):
            guard();document=docs[point['document_id']]
            if hashlib.sha256(json.dumps(document['tokens']).encode()).hexdigest()!=point['original_document_token_sha256']:raise AssertionError('Original identifier source tokens changed')
            source_views=views(document['tokens'],point);noise=initial_state(c,point['state_key'],case['state_seed'])
            losses={}
            for label,v in source_views.items():
                inputs=torch.tensor([v['inputs']]);targets=torch.tensor(v['targets']);positions=torch.tensor(v['positions'])
                for depth in [1,2]:
                    current=noise[:,v['state_start']:v['state_start']+len(v['inputs'])].clone()
                    logits=model(inputs,position_ids=positions,num_steps_pair=torch.tensor([depth-1,1]),return_logits=True)['logits'][0,v['prediction_positions']].double()
                    value=(logits.logsumexp(-1)-logits.gather(-1,targets[:,None]).squeeze(-1)).sum()
                    if not torch.isfinite(value):raise AssertionError('Nonfinite authentic-context prediction')
                    losses[label+str(depth)]=float(value)
            rows.append({'id':point['id'],'group':point['group'],'band':point['band'],
                         'real_targets':point['target_end_token']-point['target_start_token'],'nll_sum':losses})
            if (i+1)%10==0:write_state(a.output/'progress.json',{'identifiers':i+1,'wall_seconds':time.time()-started})
    if digest(dict(sorted(model.state_dict().items())))!=before:raise AssertionError('Context readout changed model')
    metrics=summarize(rows);intervals=group_intervals(rows)
    result={'status':'COMPLETE_FROZEN_NATURAL_CODE_CONTEXT_READOUT','readout':a.readout,
        'metrics':metrics,'conditional_repository_uncertainty':intervals,'primary_mechanism_readout':a.readout=='matched',
        'positive_context_use_pilot':all(intervals['intervals_95'][k][0]>0 for k in ['full_gain','interaction']),
        'panel_sha256':selection['panel_sha256'],'checkpoint_sha256':a.checkpoint_sha,'main_plan_sha256':a.plan_sha,
        'full_state_readback_exact':True,'upstream_commit':PIN,'identifiers':len(rows),
        'position_policy':'Original full-view RoPE positions and hidden-state slices retained in short views.',
        'assigned_chips':0,'optimizer_updates':0,'test_accessed':False,'new_terminal_attempt':False,
        'wall_seconds':time.time()-started,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        'scope':'One fitted candidate,consumedDEV authentic-code identifier prediction at supported exits1/2. Both full gain and interaction must have positive nominal repository lower bounds for pilot context-use finding. No general reasoning or terminal training-seed significance claim.'}
    write_state(a.output/'report.json',result);write_state(a.output/'paired-private.json',{'records':rows})
    print(json.dumps({'status':result['status'],'metrics':metrics,'seconds':time.time()-started}),flush=True)


if __name__=='__main__':main()
