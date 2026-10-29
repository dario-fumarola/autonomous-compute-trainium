"""FP64 CPU readout of every registered endpoint after a Native gate failure.

No partial Native scores are admitted and no trained parameters are changed.
The old deadline, source panels, target masks and statistical contrasts remain.
"""
import argparse,json,resource,shutil,time
from pathlib import Path

ID='probability-joint-exit-readout-recovery-v1'
PUBLIC=Path('results/probability_workspace_20261029/joint-exit-readout-recovery-v1')

def prepare(root):
    from compute_trainium.training import file_sha256
    from compute_trainium.queue import write_state
    from compute_trainium.execution_isolation import managed_running
    if managed_running():raise RuntimeError('Predecessor must be closed')
    parent=root/'probability-joint-exit-screen-v1/plan.json'
    p=json.loads(parent.read_text());failure=Path('results/probability_workspace_20261029/joint-exit-screen-v1/report.json')
    r=json.loads(failure.read_text())
    if r['status']!='STOPPED_JOINT_EXIT_SCREEN':raise AssertionError('Recorded stopped history required')
    audit=r['records']['audit-workspace']
    if audit['status']!='PASS_PROBABILITY_SCREEN_MATURE_AUDITS':raise AssertionError('Mature training audit required')
    specs=[]
    for label,depth,perm in [('joint-k0',0,False),('joint-k1',1,False),('joint-permuted-k1',1,True)]:
        path=root/'runs'/p['arms']['workspace']['phase']/'PRIVATE-u1530.pt'
        if file_sha256(path)!=audit['rows'][0]['checkpoint_sha256']:raise AssertionError('Audited candidate changed')
        specs.append(dict(label=label,depth=depth,permuted=perm,path=str(path.relative_to(root)),sha256=file_sha256(path)))
    for arm,label,depth in [('workspace','untied-one-k1',1),('ordinary_data','untied-ordinary-data-k0',0),('ordinary_work','untied-ordinary-work-k0',0)]:
        item=p['reused_arms'][arm];path=root/item['checkpoint_relative']
        if file_sha256(path)!=item['sha256']:raise AssertionError('Audited historical control changed')
        specs.append(dict(label=label,depth=depth,permuted=False,path=item['checkpoint_relative'],sha256=item['sha256']))
    sources={str(f):file_sha256(f) for d in ('compute_trainium','scripts','configs') for f in Path(d).rglob('*.py')}
    for f in ('tests/test_joint_exit_analysis.py','docs/JOINT_EXIT_READOUT_RECOVERY_20261029.md'):sources[f]=file_sha256(f)
    now=time.time();deadline=p['absolute_deadline_unix']
    if deadline-now<1200:raise RuntimeError('Recovery must fit original deadline')
    q=dict(identity=ID,registered_unix=now,absolute_deadline_unix=deadline,seconds_limit=deadline-now,
      clock_admission_waiver=True,cpu_phase_seconds_limit=min(2400,deadline-now),phase_seconds_limit=2400,
      source_sha256=sources,root_comm_ports={},parent_plan_relative=str(parent.relative_to(root)),parent_plan_sha256=file_sha256(parent),
      stopped_report_sha256=file_sha256(failure),models=specs,batch=2,threads=32,precision='FP64',
      optimizer_updates=0,assigned_chips=0,resource_memory_gib=48,
      scope='All six endpoints recomputed on CPU in FP64 with independent SDPA and CE. No threshold relaxation, source selection, new fitting, or reuse of partial Native scores. Original scientific deadline and contrasts retained. Historical and failed Native costs remain separate.')
    out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',q);write_state(PUBLIC/'plan.json',q)

def score(root,plan,output):
    import numpy as np
    import torch
    from compute_trainium.training import file_sha256
    from compute_trainium.probability_model_factory import probability_model
    from compute_trainium.probability_intervention import ProposalReadout
    from scripts.analyze_joint_exit_probability import analyze
    q=json.loads(plan.read_text());p=json.loads((root/q['parent_plan_relative']).read_text());start=time.time()
    if file_sha256(root/q['parent_plan_relative'])!=q['parent_plan_sha256']:raise AssertionError('Parent changed')
    if any(file_sha256(n)!=h for n,h in q['source_sha256'].items()):raise AssertionError('Frozen recovery source changed')
    torch.set_num_threads(q['threads']);output.mkdir(exist_ok=False)
    folder=root/p['evaluation_relative'];receipt=json.loads((folder/'report.json').read_text())
    if file_sha256(folder/'report.json')!=p['evaluation_report_sha256'] or any(file_sha256(folder/n)!=h for n,h in receipt['files'].items()):raise AssertionError('Frozen evaluation bytes changed')
    broad=np.load(folder/'broad-tokens.npy',mmap_mode='r')
    panels={'broad':(broad[:,:-1],broad[:,1:],np.ones((len(broad),512),dtype=np.bool_),json.loads((folder/'PRIVATE-broad-metadata.json').read_text())),
      'dependency':(np.load(folder/'dependency-inputs.npy',mmap_mode='r'),np.load(folder/'dependency-targets.npy',mmap_mode='r'),np.load(folder/'dependency-mask.npy',mmap_mode='r'),json.loads((folder/'PRIVATE-dependency-metadata.json').read_text()))}
    permutation=torch.randperm(p['model_config']['vocab_size'],generator=torch.Generator().manual_seed(p['permutation_seed']))
    rows=[]
    for spec in q['models']:
        path=root/spec['path']
        if file_sha256(path)!=spec['sha256']:raise AssertionError('Frozen endpoint changed')
        saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        model=probability_model(p).double().eval();model.load_state_dict(saved['model']);del saved
        reader=ProposalReadout(model,spec['depth'],'expected',permutation if spec['permuted'] else None)
        for panel,(xs,ys,ms,meta) in panels.items():
            for left in range(0,len(xs),q['batch']):
                end=min(left+q['batch'],len(xs));x=torch.from_numpy(np.asarray(xs[left:end],dtype=np.int64));y=torch.from_numpy(np.asarray(ys[left:end],dtype=np.int64))
                masks=torch.from_numpy(np.asarray(ms[left:end],dtype=np.bool_))
                with torch.no_grad():
                    logits=reader(x)
                    losses=torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').view_as(y)
                if not torch.isfinite(losses).all():raise AssertionError('Nonfinite reference readout')
                for j,i in enumerate(range(left,end)):
                    item=meta[i];selected=losses[j][masks[j]]
                    rows.append(dict(label=spec['label'],panel=panel,index=i,id=item['id'],group=item['group'],kind=item['kind'],domain=item.get('domain'),cohort=item.get('cohort'),loss_sum=float(selected.sum()),targets=len(selected),first_loss=float(selected[0])))
                del logits,losses
                if left%32==0:(output/'progress.json').write_text(json.dumps(dict(label=spec['label'],panel=panel,left=left,seconds=time.time()-start)))
                available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
                if time.time()>q['absolute_deadline_unix'] or time.time()-start>q['cpu_phase_seconds_limit'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>48*1024**3 or available<512*1024**3 or min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*1024**3:raise RuntimeError('Frozen CPU readout resource bound')
        del reader,model
        (output/'PRIVATE-scores.json').write_text(json.dumps(rows))
    result=dict(status='COMPLETE_FP64_JOINT_EXIT_RECOVERY',**analyze(rows,p),precision='FP64',assigned_chips=0,optimizer_updates=0,
      trained_seeds=1,new_fits=0,protected_groups_scored=0,automatic_promotion=False,all_six_models_recomputed=True,
      maximum_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,wall_seconds=time.time()-start,
      scope=q['scope'])
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')

def run(root):
    from scripts.run_scale_sharding_fixture import AllChipExecution
    from compute_trainium.queue import write_state
    out=root/ID;e=AllChipExecution(root,out/'plan.json',out/'execution');r={'status':'STOPPED_FP64_JOINT_EXIT_RECOVERY'}
    try:
        r=e.cpu(ID+'-cpu','scripts.recover_joint_exit_readout',['--root',str(root),'--action','score','--plan',str(out/'plan.json'),'--output',str(out/'readout')],out/'readout/report.json','COMPLETE_FP64_JOINT_EXIT_RECOVERY')
    finally:write_state(PUBLIC/'report.json',r);e.close(PUBLIC/'execution-costs.json')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--action',choices=['prepare','run','score'],required=True);a.add_argument('--plan',type=Path);a.add_argument('--output',type=Path);p=a.parse_args()
    if p.action=='score':score(p.root,p.plan,p.output)
    else:(prepare if p.action=='prepare' else run)(p.root)
