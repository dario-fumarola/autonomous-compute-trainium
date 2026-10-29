"""TRAIN-only channel gradients: output prediction versus internal workspace."""
import argparse,json,resource,shutil,time
from pathlib import Path

def main(root):
    import torch
    from compute_trainium.model import ModelConfig
    from compute_trainium.probability_workspace import ProbabilityWorkspace
    from compute_trainium.probability_workspace_data import ProbabilitySources
    from compute_trainium.training import file_sha256
    start=time.time();torch.set_num_threads(8);out=root/'probability-channel-gradients-v1';out.mkdir(exist_ok=False)
    pf=root/'probability-workspace-screen-recovery-v1/plan.json';p=json.loads(pf.read_text());parent=root/'runs'/p['arms']['workspace']['phase']
    indices=list(range(1_050_000,1_050_012));plan=dict(registered_unix=start,seconds_limit=300,threads=8,
        source_sha256=file_sha256(__file__),screen_plan_sha256=file_sha256(pf),training_windows=indices,
        checkpoints={str(u):file_sha256(parent/f'PRIVATE-u{u}.pt') for u in (492,1530)},
        scope='Fixed TRAIN diagnostic only. Compare derivative of ordinary NTP with derivative of one-replay NTP with respect to the SAME first-pass logits. Reader parameters frozen,proposal differentiable. No optimizer updates,held-out reads,threshold changes or automatic promotion.')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');data=ProbabilitySources(root/p['data_relative']);c=ModelConfig(**p['model_config']);rows=[]
    for update in (0,492,1530):
        torch.manual_seed(p['seed']);model=ProbabilityWorkspace(c)
        if update:
            state=torch.load(parent/f'PRIVATE-u{update}.pt',map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(state['model']);del state
        model.requires_grad_(False);dot=norm0=norm1=0.;positions=opposed=0;cosines=[];losses=[0.,0.];residual=0.
        for index in indices:
            x,y,mask=data.batch([index]);count=int(mask.sum())
            with torch.no_grad():embedding=model.embedding(x);logits0=torch.nn.functional.linear(model.body(embedding),model.embedding.weight)
            channel=logits0.detach().requires_grad_(True);proposal=channel.softmax(-1)@model.embedding.weight
            logits1=torch.nn.functional.linear(model.body(embedding+proposal),model.embedding.weight)
            losses0=torch.nn.functional.cross_entropy(channel.flatten(0,1),y.flatten(),reduction='none').view_as(y)
            losses1=torch.nn.functional.cross_entropy(logits1.flatten(0,1),y.flatten(),reduction='none').view_as(y)
            loss0=(losses0*mask).sum()/count;loss1=(losses1*mask).sum()/count
            g0=torch.autograd.grad(loss0,channel,retain_graph=True)[0]
            g1=torch.autograd.grad(loss1,channel)[0]
            with torch.no_grad():
                # Independently check ordinary CE derivative;exclude synthetic
                # BOS/EOS targets from both losses and the comparison.
                analytic=channel.softmax(-1);analytic.scatter_add_(-1,y.unsqueeze(-1),-torch.ones_like(y,dtype=torch.float32).unsqueeze(-1));analytic*=mask.unsqueeze(-1)/count
                residual=max(residual,float((g0-analytic).abs().max()))
                a=(g0*count)[mask].double();b=(g1*count)[mask].double();dots=(a*b).sum(-1);aa=a.square().sum(-1);bb=b.square().sum(-1)
                dot+=float(dots.sum());norm0+=float(aa.sum());norm1+=float(bb.sum());positions+=len(dots);opposed+=int((dots<0).sum())
                cosines.extend((dots/(aa*bb).sqrt().clamp_min(1e-30)).tolist());losses[0]+=float(loss0)*count;losses[1]+=float(loss1)*count
            del logits1,proposal,channel,g0,g1,a,b,analytic,loss0,loss1,losses0,losses1
            available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
            if time.time()-start>300 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>48*1024**3 or available<512*1024**3 or shutil.disk_usage(root).free<35*1024**3:raise RuntimeError('Channel diagnostic resource bound')
        if residual>1e-6:raise AssertionError('Independent CE derivative check')
        rows.append(dict(update=update,real_train_positions=positions,global_channel_cosine=dot/(norm0*norm1)**.5,
            mean_position_cosine=sum(cosines)/len(cosines),opposed_position_fraction=opposed/positions,
            replay_to_direct_gradient_norm_ratio=(norm1/norm0)**.5,maximum_ce_derivative_error=residual,
            train_zero_replay_nll=losses[0]/positions,train_one_replay_nll=losses[1]/positions))
        del model
    result=dict(status='COMPLETE_TRAIN_CHANNEL_GRADIENT_DIAGNOSTIC',rows=rows,wall_seconds=time.time()-start,assigned_chips=0,optimizer_updates=0,heldout_reads=0,
        interpretation='Same-logit local derivatives on12 fixed TRAIN windows. Negative alignment would show competing local channel directions here,not prove a cause of optimization or transfer failure. Positive alignment does not prove useful reasoning. These losses are TRAIN diagnostics,not development or confirmation scores.')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);main(parser.parse_args().root)
