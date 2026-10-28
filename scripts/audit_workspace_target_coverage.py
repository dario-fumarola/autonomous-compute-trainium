"""TRAIN-only lower bound on explicit dependencies in the old branch screen."""
import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import time


def main(root, output):
    import torch
    from transformers import AutoTokenizer
    from compute_trainium.workspace_view_batches import WorkspaceViewBatches
    from compute_trainium.source_arithmetic import original_chains
    from compute_trainium.natural_code_context import unambiguous_local_uses, token_span
    from compute_trainium.training import file_sha256
    output.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4)
    source=root/'workspace-view-batches-v1';data=WorkspaceViewBatches(source,'train')
    tokenizer=AutoTokenizer.from_pretrained(root/'workspace-olmo2-stage1-base-v1',local_files_only=True)
    plan=dict(scope='TRAIN only; no model scores or source content published',views=8192,
              sampled_targets_per_view=32,total_possible_targets_per_view=159,query_rng_seed=7286,
              batch_views=4,source_sha256={p.name:file_sha256(p) for p in sorted(source.glob('train-*.npy'))},
              parser_scope='Explicit verified arithmetic chains and strict whole-fragment Python AST local bindings; lower bounds only')
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    rng=torch.Generator().manual_seed(7286);counts=Counter();bydomain={str(i):Counter() for i in range(3)}
    for batch in range(2048):
        selected=set(torch.randperm(159,generator=rng)[:32].tolist())
        for index in range(batch*4,batch*4+4):
            row=data.at(index);ids=torch.cat([row['prefix'],row['continuation']],1).flatten().tolist()
            text=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
            own=Counter(views=1,selected_targets=32,natural_targets=159)
            if encoded['input_ids']!=ids:
                own['decode_reencode_mismatch']=1
            else:
                own['exact_token_roundtrip']=1;spans={}
                chains=list(original_chains(text));own['arithmetic_chains']=len(chains)
                spans['arithmetic_conclusion']=[c['conclusion'] for c in chains]
                spans['arithmetic_intermediate']=[s for c in chains for s in c['intermediate_steps']]
                try:
                    code=unambiguous_local_uses(text);own['parseable_python_fragments']=1
                except (SyntaxError,ValueError,RecursionError):
                    code=[];own['unparseable_python_fragments']=1
                own['unambiguous_code_uses']=len(code)
                spans['code_identifier']=[(c['target_start'],c['target_end']) for c in code]
                for kind,items in spans.items():
                    targets=set()
                    for a,b in items:
                        mapped=token_span(text,encoded['offset_mapping'],a,b)
                        if mapped:
                            targets.update(i-1 for i in range(*mapped) if 1<=i<160)
                        else:own[kind+'_unmapped_spans']+=1
                    own[kind+'_natural_targets']=len(targets)
                    own[kind+'_selected_targets']=len(targets&selected)
                    own[kind+'_views_with_selected_target']=int(bool(targets&selected))
            counts.update(own);bydomain[str(row['domain'])].update(own)
        if batch%128==0:
            (output/'progress.json').write_text(json.dumps(dict(completed_views=(batch+1)*4,seconds=time.time()-start)))
        if time.time()-start>600 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>48*1024**3:
            raise RuntimeError('Finite 600-second/48GiB TRAIN audit bound')
    saved=torch.load(root/'residual-data-volume-v1/PRIVATE-fresh_one-state.pt',weights_only=True,mmap=True)
    if not torch.equal(saved['rng'],rng.get_state()):raise AssertionError('Does not reproduce actual trained target RNG')
    result=dict(status='COMPLETE_TRAIN_DEPENDENCY_COVERAGE',counts=dict(counts),by_domain={k:dict(v) for k,v in bydomain.items()},
        exact_endpoint_target_rng_match=True,model_scores=0,held_out_reads=0,optimizer_updates=0,assigned_chips=0,
        wall_seconds=time.time()-start,scope=plan['parser_scope'],
        limitation='Nonmatches include truncated snippets and dependencies outside these strict rules; absence is not evidence of no computation opportunity.')
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.root,a.output)
