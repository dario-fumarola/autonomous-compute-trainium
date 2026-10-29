"""TRAIN-only descriptive proposal statistics; no fitting or quality selection."""
import argparse,json,resource,time
from pathlib import Path
import torch
from compute_trainium.probability_model_factory import probability_model
from compute_trainium.probability_workspace_data import ProbabilitySources
from compute_trainium.training import file_sha256

def stats(prob,thought,state):
    p=prob.double();t=thought.double();s=state.double();mean=t.mean((0,1));variation=((t-mean).square().sum(-1).mean()).sqrt();rms=t.square().sum(-1).mean().sqrt()
    return dict(entropy_nats=float(-(p*p.clamp_min(1e-300).log()).sum(-1).mean()),top1_mass=float(p.max(-1).values.mean()),
      thought_rms_l2=float(rms),thought_centered_rms_l2=float(variation),thought_centered_to_total=float(variation/rms.clamp_min(1e-30)),
      state_rms_l2=float(s.square().sum(-1).mean().sqrt()),mean_thought_l2=float(mean.norm()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(16);start=time.time();rows=[];ids=['probability-fixed-one-v1','probability-untied-screen-v1']
    # Fixed-one plan retains original sampled control as a frozen arm.
    for identity,armname,depths in [(ids[0],'workspace',[1]),(ids[1],'workspace',[1])]:
        planpath=a.root/identity/'plan.json';plan=json.loads(planpath.read_text());cfg=plan['arms'][armname]
        audit=a.root/identity/('audit-'+armname)/'report.json'
        if json.loads(audit.read_text())['status']!='PASS_PROBABILITY_SCREEN_MATURE_AUDITS':raise AssertionError('Mature audit required')
        ck=a.root/'runs'/cfg['phase']/f"PRIVATE-u{cfg['checkpoint_updates'][-1]}.pt"
        saved=torch.load(ck,map_location='cpu',weights_only=False,mmap=True);model=probability_model(plan);model.load_state_dict(saved['model']);model.eval();del saved
        data=ProbabilitySources(a.root/plan['data_relative']);x,_,_=data.batch(list(range(plan['audit_window_start'],plan['audit_window_start']+8)))
        records=[]
        with torch.no_grad():
            for depth in depths:
                state=model.embedding(x);steps=[]
                for remaining in range(depth,0,-1):
                    z=state
                    if hasattr(model,'remaining_budget_cues'):z=z+model.remaining_budget_cues[remaining-1]
                    output=getattr(model,'prediction_weight',model.embedding.weight)
                    logits=model.body(z)@output.T;prob=logits.softmax(-1);thought=prob@model.embedding.weight
                    steps.append(dict(remaining=remaining,**stats(prob,thought,state)));state=state+thought
                    del logits,prob,thought
                records.append(dict(extra_steps=depth,steps=steps))
        table_stats={}
        if hasattr(model,'prediction_weight'):
            e=model.embedding.weight.detach().double();w=model.prediction_weight.detach().double()
            table_stats=dict(relative_output_input_distance=float((w-e).norm()/e.norm()),table_cosine=float((w*e).sum()/(w.norm()*e.norm())),input_l2=float(e.norm()),output_l2=float(w.norm()))
            del e,w
        rows.append(dict(identity=identity,arm=armname,table_stats=table_stats,checkpoint_sha256=file_sha256(ck),plan_sha256=file_sha256(planpath),audit_sha256=file_sha256(audit),train_window_start=plan['audit_window_start'],windows=8,records=records))
        del model,data,x,state
    result=dict(status='COMPLETE_UNTIED_TRAIN_PROPOSAL_STATISTICS',rows=rows,optimizer_updates=0,heldout_reads=0,assigned_chips=0,wall_seconds=time.time()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
      scope='Descriptive proposal entropy and state variation on the same first8 TRAIN audit windows. These are not held-out scores,causal ablations or proof of a reasoning mechanism. No new sampled thoughts or future-target access;all reported depths supported. The untied model uses its independent output table for logits and its input table for proposal values. Table distance is descriptive,not a performance or causal attribution test. Global constant mean is descriptive,not a substitute control.')
    if result['peak_rss_bytes']>48*1024**3 or result['wall_seconds']>300:raise RuntimeError('Resource bound')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
