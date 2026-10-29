"""Bounded TRAIN-only proposal geometry; no target scoring or model selection."""
import argparse,json,resource,shutil,time
from pathlib import Path

def main(root):
    import torch
    from compute_trainium.model import ModelConfig
    from compute_trainium.probability_workspace import ProbabilityWorkspace,expected_embedding
    from compute_trainium.probability_workspace_data import ProbabilitySources
    from compute_trainium.training import file_sha256
    from scripts.qualify_native import digest
    start=time.time();torch.set_num_threads(8)
    out=root/'probability-proposal-geometry-v1';out.mkdir(exist_ok=False)
    planfile=root/'probability-workspace-screen-recovery-v1/plan.json';p=json.loads(planfile.read_text())
    parent=root/'runs'/p['arms']['workspace']['phase'];indices=list(range(1_050_000,1_050_012))
    plan=dict(registered_unix=start,seconds_limit=300,threads=8,max_rss_bytes=48*1024**3,
        screen_plan_sha256=file_sha256(planfile),source_sha256=file_sha256(__file__),training_windows=indices,
        checkpoints={str(u):file_sha256(parent/f'PRIVATE-u{u}.pt') for u in (492,1530)},
        scope='No target labels read,loss/reward evaluated,held-out sources read,model selection,updates or new Native work. Fixed12 authentic TRAIN windows outside fitted prefixes;raw proposal geometry only. Does not change the frozen comparison or permit promotion.')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');data=ProbabilitySources(root/p['data_relative']);c=ModelConfig(**p['model_config']);rows=[]
    for update in (0,492,1530):
        torch.manual_seed(p['seed']);m=ProbabilityWorkspace(c)
        if update:
            checkpoint=torch.load(parent/f'PRIVATE-u{update}.pt',map_location='cpu',weights_only=False,mmap=True);m.load_state_dict(checkpoint['model']);del checkpoint
        else:
            if digest(m.state_dict())!=json.loads((parent/'report.json').read_text())['initial_model_digest']:raise AssertionError('Initial state differs')
        sums=[dict(entropy=0.,top_probability=0.,proposal_norm=0.,input_norm=0.,centered_proposal_norm=0.,positions=0) for _ in (1,2)]
        for index in indices:
            # Read only model inputs,not the next-token target/mask.
            x=torch.from_numpy(data.tokens[index*512:(index+1)*512].astype('int64')).unsqueeze(0)
            with torch.no_grad():
                base=m.embedding(x);state=base
                for depth in range(2):
                    logits=torch.nn.functional.linear(m.body(state),m.embedding.weight);logp=logits.log_softmax(-1);prob=logp.exp();proposal=prob@m.embedding.weight
                    r=sums[depth];r['entropy']+=float(-(prob*logp).sum());r['top_probability']+=float(prob.amax(-1).sum())
                    r['proposal_norm']+=float(proposal.norm(dim=-1).sum());r['input_norm']+=float(base.norm(dim=-1).sum())
                    r['centered_proposal_norm']+=float((proposal-m.embedding.weight.mean(0)).norm(dim=-1).sum());r['positions']+=x.numel();state=state+proposal
            available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
            if time.time()-start>300 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>48*1024**3 or available<512*1024**3 or shutil.disk_usage(root).free<35*1024**3:raise RuntimeError('TRAIN geometry resource bound')
        for depth,s in enumerate(sums,1):
            row=dict(update=update,proposal_step=depth,positions=s['positions'],**{k:v/s['positions'] for k,v in s.items() if k!='positions'})
            row['proposal_to_input_norm_ratio']=s['proposal_norm']/s['input_norm'];rows.append(row)
        del m
    result=dict(status='COMPLETE_TRAIN_ONLY_PROPOSAL_GEOMETRY',rows=rows,assigned_chips=0,optimizer_updates=0,target_loss_scores=0,heldout_reads=0,
        wall_seconds=time.time()-start,scope='Descriptive proposal geometry only. Greater magnitude/confidence does not establish correctness,reasoning or useful extra computation.')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);main(parser.parse_args().root)
