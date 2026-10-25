"""Read-only prefix-only original-answer baseline,with uncached generation."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
os.environ['TORCH_NEURONX_ENABLE_NKI_SDPA']='0'
os.environ['NEURON_RT_VISIBLE_CORES']=os.environ['RESEARCH_VISIBLE_CORES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))]
import numpy as np
import torch
import torch.distributed as dist
from tokenizers import Tokenizer
from compute_trainium.conclusion_generation import GreedyConclusion,correct_scalar
from compute_trainium.launcher_compat import without_launcher_corpus
from compute_trainium.metrics import real_token_losses
from compute_trainium.parcae_candidate import ParcaeCandidateConfig,matmul_ledger
from compute_trainium.parcae_primitive_sdpa_candidate import ParcaePrimitiveSDPACandidate
from compute_trainium.training import file_sha256
from scripts.qualify_native import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True)
    p.add_argument('--case',required=True);p.add_argument('--comm-port',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(without_launcher_corpus(sys.argv[1:]));started=time.time()
    if file_sha256(a.plan)!=a.plan_sha:raise AssertionError('Readout plan changed')
    plan=json.loads(a.plan.read_text());case=plan['cases'][a.case];code=Path(__file__).resolve().parents[1]
    if plan['purpose']!='readonly_original_chain_baseline' or plan['root_comm_ports'][a.output.name]!=a.comm_port or any(file_sha256(code/f)!=h for f,h in plan['worker_source_sha256'].items()):raise ValueError('Frozen read-only original-answer plan required')
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(a.comm_port)
    import torch_neuronx
    torch.set_num_threads(2);torch.neuron.set_device(0);dist.init_process_group('neuron');rank=dist.get_rank();world=dist.get_world_size()
    if world!=32:raise ValueError('32 read-only replicas required')
    a.output.mkdir(exist_ok=True);checkpoint=a.root/case['checkpoint'];panel_file=a.root/plan['panel'];noise_file=a.root/plan['noise']
    if rank==0:
        for f,h in [(checkpoint,case['checkpoint_sha256']),(panel_file,plan['panel_sha256']),(noise_file,plan['noise_sha256'])]:
            if file_sha256(f)!=h:raise AssertionError('Read-only input changed')
    dist.barrier();cfg=ParcaeCandidateConfig(**plan['model']);saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    state={k:v for k,v in saved['model'].items() if not k.startswith('future_lm_head.') and k!='future_positions'}
    model=ParcaePrimitiveSDPACandidate(cfg,recurrent=True,initialize=False,fp16_body=True);model.load_state_dict(state,strict=True);del state,saved
    model.eval();initial_digest=digest(dict(sorted(model.state_dict().items())));model.to('neuron');model.compile_stages('neuron')
    tokenizer=Tokenizer.from_file('/work/data-v1/tokenizer.json');panel=json.loads(panel_file.read_text());noise=np.load(noise_file,mmap_mode='r');records=[]
    with torch.no_grad():
        for index in range(rank,len(panel),32):
            row=panel[index];view=row['views'][0]
            if view['visible_steps']!=0:raise ValueError('Only the no-work inference view is eligible')
            h0=torch.from_numpy(np.array(noise[index:index+1],copy=True)).to('neuron')
            for n in case['depths']:
                run=GreedyConclusion(tuple(view['prefix_ids']),context=128,maximum_new_tokens=24);calls=0;first=None;t0=time.time()
                while not run.done:
                    ids=torch.tensor(run.input_ids(),dtype=torch.long)[None].to('neuron');position=run.prediction_position
                    logits=model(ids,h0,total_steps=n,backward_cap=n)
                    pos=torch.tensor([[[position]]],dtype=torch.long).to('neuron').expand(1,1,cfg.vocab_size)
                    token=int(logits.gather(1,pos).argmax(-1).cpu().item());calls+=1
                    if first is None:first=token
                    run.advance(token)
                decoded=tokenizer.decode(run.generated,skip_special_tokens=False);elapsed=time.time()-t0
                # Gold enters only this separate teacher-forced scoring call.
                inputs=torch.tensor(view['inputs'],dtype=torch.long)[None].to('neuron');targets=torch.tensor(view['targets'],dtype=torch.long)[None].to('neuron');mask=torch.tensor(view['answer_mask'],dtype=torch.bool)[None].to('neuron')
                logits=model(inputs,h0,total_steps=n,backward_cap=n);loss=float(real_token_losses(logits,targets,mask).sum().cpu())
                records.append(dict(document_id=row['id'],group=row['group'],depth=n,original_answer_tokens=sum(view['answer_mask']),nll_sum=loss,
                    correct=correct_scalar(decoded,view['target'],run.ended),terminated=run.ended,first_token_correct=first==view['answer_ids'][0],
                    generated_tokens=len(run.generated),forward_calls=calls+1,generation_seconds=elapsed,prediction=decoded))
    final_digest=digest(dict(sorted(model.state_dict().items())));fallback=list(torch_neuronx.get_fallback_ops())
    if final_digest!=initial_digest or set(fallback)-{'aten::_to_copy'}:raise AssertionError('Read-only weights changed or unsupported fallback')
    (a.output/f'records-rank{rank}.json').write_text(json.dumps(records)+'\n');(a.output/f'state-rank{rank}.json').write_text(json.dumps(dict(initial=initial_digest,final=final_digest,fallback_ops=fallback))+'\n');dist.barrier()
    if rank==0:
        all_rows=[r for i in range(32) for r in json.loads((a.output/f'records-rank{i}.json').read_text())]
        states=[json.loads((a.output/f'state-rank{i}.json').read_text()) for i in range(32)]
        if len({r['initial'] for r in states})!=1:raise AssertionError('Baseline replicas differ')
        summaries={};work=0
        for n in case['depths']:
            rows=[r for r in all_rows if r['depth']==n]
            if {r['document_id'] for r in rows}!={r['id'] for r in panel} or len(rows)!=len(panel):raise AssertionError('Readout coverage differs')
            count=sum(r['original_answer_tokens'] for r in rows);calls=sum(r['forward_calls'] for r in rows);work+=calls*matmul_ledger(cfg,128,total_steps=n,backward_cap=n)['forward_matmul_flops']
            summaries[str(n)]=dict(tasks=len(rows),answer_tokens=count,answer_nll=sum(r['nll_sum'] for r in rows)/count,correct=sum(r['correct'] for r in rows),terminated=sum(r['terminated'] for r in rows),first_token_correct=sum(r['first_token_correct'] for r in rows),generated_tokens=sum(r['generated_tokens'] for r in rows),forward_calls=calls,summed_generation_seconds_across_replicas=sum(r['generation_seconds'] for r in rows))
        (a.output/'records.json').write_text(json.dumps(sorted(all_rows,key=lambda r:(r['depth'],r['document_id'])))+'\n')
        result=dict(status='COMPLETE_READONLY_ORIGINAL_CHAIN_BASELINE',case=a.case,plan_sha256=a.plan_sha,checkpoint_sha256=case['checkpoint_sha256'],depths=summaries,
            records_sha256=file_sha256(a.output/'records.json'),weights_unchanged=True,all_replicas_identical=True,nominal_forward_matrix_ops=work,
            optimizer_updates=0,research_optimizer_updates=0,test_accessed=False,wall_seconds=time.time()-started,
            scope='New discovery DEV only;prefix-only greedy original scalar conclusions,EOS required,24-token fixed budget. Teacher-forced answer loss scored afterward. Full128-position recomputation,no KV cache or serving-efficiency claim. No fit or terminal confirmation.')
        (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
