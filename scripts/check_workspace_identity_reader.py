"""Framing-free reader;zero work is exactly the original prefix."""
import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from compute_trainium.training import file_sha256
from compute_trainium.workspace_learning import (Reader,ForwardLedger,WorkspaceFormat,
    sample_workspaces,score_workspaces,framed_context,identity_reader_context)

PUBLIC=Path('results/workspace_identity_reader_20261027/recovery-v2')
DOCUMENT='docs/WORKSPACE_IDENTITY_READER_20261027.md'


def run(root):
    started=time.time();out=root/'workspace-identity-reader-v2';bootstrap=root/'workspace-reader-bootstrap-v1'
    old=root/'workspace-reader-qualification-v1';base=root/'workspace-olmo2-stage1-base-v1'
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every declared interface check')
    proof=json.loads((bootstrap/'report.json').read_text());initial=json.loads((old/'report.json').read_text())
    if proof['status']!='COMPLETE_BOUNDED_READER_BOOTSTRAP':raise ValueError('Complete calibrated checkpoint required')
    if file_sha256(bootstrap/'checkpoint.pt')!=proof['checkpoint_sha256']:raise AssertionError('Calibrated weights changed')
    views=json.loads((old/'views.json').read_text());format_plan=json.loads((old/'plan.json').read_text())
    fmt=WorkspaceFormat(tuple(format_plan['opening_ids']),tuple(format_plan['closing_ids']))
    out.mkdir(mode=0o700);PUBLIC.mkdir();torch.set_num_threads(16)
    plan=dict(registered_unix=time.time(),checkpoint_sha256=proof['checkpoint_sha256'],source_views_sha256=file_sha256(old/'views.json'),
        source_sha256={f:file_sha256(f) for f in ['scripts/check_workspace_identity_reader.py','compute_trainium/workspace_learning.py',DOCUMENT]},
        group_size=4,length=8,horizon=4,seed=204137,maximum_seconds=900,optimizer_updates=0,assigned_chips=0,
        empty_frame_tolerance=.05,privileged_copy_roomy_gain=.1,roomy_indices=[i for i,v in enumerate(initial['measurements']) if v['raw_nll']>=.1],
        scope=Path(DOCUMENT).read_text())
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');(PUBLIC/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    result=dict(status='STOPPED_IDENTITY_READER_CHECK',optimizer_updates=0,assigned_chips=0);ledger=ForwardLedger()
    try:
        tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
        pause_id=tokenizer.encode('.',add_special_tokens=False)[0]
        model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,
            dtype=torch.float32,attn_implementation='eager').eval().requires_grad_(False)
        state=torch.load(bootstrap/'checkpoint.pt',weights_only=True,mmap=True);model.load_state_dict(state['model']);del state
        versions=tuple(p._version for p in model.parameters());reader=Reader(lambda x:model(input_ids=x,use_cache=False).logits,ledger)
        samples=[]
        for i,v in enumerate(views):
            samples.append(sample_workspaces(reader,torch.tensor([v['prefix']]),format=fmt,length=8,group_size=4,
                generator=torch.Generator().manual_seed(204137+i)))
        measurements=[]
        for i,(v,z) in enumerate(zip(views,samples)):
            if time.time()-started>900:raise TimeoutError('Bounded reader-order check ended')
            available=next(int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:'))
            if available<528*1024**3:raise RuntimeError('Protected host plus CPU headroom required')
            (out/'progress.json').write_text(json.dumps(dict(anchor=i,wall_seconds=time.time()-started)))
            c=torch.tensor([v['prefix']]);y=torch.tensor([v['targets']]);row=dict(domain=v['domain'])
            pause=torch.full((1,8),pause_id,dtype=torch.long)
            for label,builder in [('identity',identity_reader_context)]:
                own=score_workspaces(reader,c,y,z,format=fmt,context_builder=builder)
                donor=score_workspaces(reader,c,y,samples[(i+1)%len(samples)][:1],format=fmt,context_builder=builder)
                fixed=score_workspaces(reader,c,y,pause,format=fmt,context_builder=builder)
                privileged=score_workspaces(reader,c,y,y.repeat(1,2),format=fmt,context_builder=builder)
                row[label]=dict(raw_nll=-float(own['raw_log_probability'].mean()),empty_nll=-float(own['framed_log_probability'].mean()),
                    own_nll=-float(own['workspace_log_probability'].mean()),donor_nll=-float(donor['workspace_log_probability'].mean()),
                    pause_nll=-float(fixed['workspace_log_probability'].mean()),privileged_copy_gain=float(privileged['absolute_gain'].mean()),
                    own_gain=float(own['absolute_gain'].mean()),reward_std=float(own['absolute_gain'].std()))
            measurements.append(row)
        candidate=[v['identity'] for v in measurements]
        empty=sum(v['empty_nll']-v['raw_nll'] for v in candidate)/len(candidate)
        privileged=sum(candidate[i]['privileged_copy_gain'] for i in plan['roomy_indices'])/len(plan['roomy_indices'])
        if tuple(p._version for p in model.parameters())!=versions:raise AssertionError('Readout changed weights')
        result.update(status='COMPLETE_IDENTITY_READER_CHECK',measurements=measurements,empty_frame_penalty=empty,
            privileged_copy_gain_on_roomy_anchors=privileged,empty_frame_gate=empty<=.05,privileged_copy_gate=privileged>=.1,
            both_point_diagnostic_gates_pass=empty<=.05 and privileged>=.1,
            warning='Previously consumed TRAIN diagnostics,zero updates. Passing interface gates is not a thinking or efficiency result.')
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        result.update(wall_seconds=time.time()-started,forward_ledger=ledger.summary(),plan_sha256=file_sha256(out/'plan.json'))
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');(PUBLIC/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
