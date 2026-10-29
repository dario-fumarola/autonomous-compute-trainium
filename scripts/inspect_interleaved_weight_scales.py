"""Exploratory aggregate parameter scales, no source examples or optimization."""
import argparse
import json
import math
import time
from pathlib import Path


def main(root, output):
    import torch
    from compute_trainium.interleaved_model_factory import make_model
    from compute_trainium.training import file_sha256
    start=time.time();torch.set_num_threads(8)
    plan_path=root/'interleaved-lr-tuning-v1/plan.json';p=json.loads(plan_path.read_text())
    torch.manual_seed(p['seed']);initial=make_model(p).state_dict();rows=[]
    for arm in ['initialization','latent-lr0','latent-lr1','latent-lr2']:
        checkpoint_hash=None
        if arm=='initialization':state=initial
        else:
            a=p['arms'][arm];update=a['checkpoint_updates'][-1]
            receipt=json.loads((root/'interleaved-lr-tuning-v1'/('audit-'+arm)/'report.json').read_text())
            if receipt['status']!='PASS_INTERLEAVED_SCREEN_MATURE_AUDITS':raise AssertionError('Complete audited weights only')
            checkpoint=root/'runs'/a['phase']/f'PRIVATE-u{update}.pt'
            checkpoint_hash=file_sha256(checkpoint)
            if checkpoint_hash!=receipt['rows'][-1]['checkpoint_sha256']:raise AssertionError('Audited bytes changed')
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);state=saved['model']
        stats=dict(arm=arm,checkpoint_sha256=checkpoint_hash,
            source_embedding_parameter_rms=float(state['embedding.weight'].float().square().mean().sqrt())*math.sqrt(2.5*p['model_config']['dim']),
            final_norm_parameter_rms=float(state['norm.weight'].float().square().mean().sqrt()))
        for ending in ['q.weight','k.weight','v.weight','o.weight','up.weight','down.weight']:
            values=[v for n,v in state.items() if n.startswith('blocks.') and n.endswith('.'+ending)]
            stats[ending+'_rms']=math.sqrt(sum(float(v.float().square().sum()) for v in values)/sum(v.numel() for v in values))
        rows.append(stats)
        if arm!='initialization':del saved,state
    result=dict(status='COMPLETE_EXPLORATORY_PARAMETER_SCALE_INSPECTION',rows=rows,
        training_plan_sha256=file_sha256(plan_path),source_sha256=file_sha256(Path(__file__)),
        assigned_chips=0,optimizer_updates=0,source_example_reads=0,wall_seconds=time.time()-start,
        scope='Post-diagnostic descriptive parameter norms. Final norm parameter RMS is not a measured latent activation RMS. No causal explanation, prediction quality, allocation or new fit claim.')
    output.mkdir(exist_ok=False);(output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.root,a.output)
