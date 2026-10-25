"""After a completed negative pilot,measure the frozen models on TRAIN only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
from compute_trainium.future_span_execution import Execution
from compute_trainium.inode_reserve import require_inode_reserve
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256
from scripts import run_natural_chain_baseline as original

IDENTITY='natural-chain-train-diagnostic-v1'
PUBLIC=Path('results/natural_chain_train_diagnostic_20261025')

class ScopedExecution(Execution):
    def check(self):super().check();require_inode_reserve([self.root,Path('/')])
    def native(self,identity,module,options,status):
        if module=='scripts.read_natural_chain_native':module='scripts.read_natural_chain_train_native'
        return super().native(identity,module,options,status)
    def cpu(self,identity,module,options,report_file,status):
        if module=='scripts.audit_natural_chain_readout_cpu':module='scripts.audit_natural_chain_train_cpu'
        return super().cpu(identity,module,options,report_file,status)


def main():
    import numpy as np
    from compute_trainium.parcae_candidate import ParcaeCandidateConfig
    from compute_trainium.parcae_training_candidate import initial_state
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;prior=root/'natural-chain-continuation-pilot-v2'
    completed=json.loads((prior/'report.json').read_text())
    if completed['status']!='COMPLETE_NATURAL_CHAIN_CONTINUATION_PILOT':raise ValueError('Completed fixed pilot required')
    old=json.loads((prior/'evaluation-plan.json').read_text());master=json.loads((prior/'plan.json').read_text());source=root/'natural-chain-views-v1/training-views.json'
    if file_sha256(source)!=master['source_views_sha256']:raise AssertionError('Original TRAIN views changed')
    views=json.loads(source.read_text());grouped={}
    for row in views:grouped.setdefault(row['document_id'],[]).append(row)
    selected=[];groups={}
    for key in sorted(grouped,key=lambda k:hashlib.sha256(('chain-train-diagnostic-v1:'+k).encode()).hexdigest()):
        row=next(r for r in grouped[key] if r['visible_steps']==0);group=row['group']
        if groups.get(group,0)>=2:continue
        selected.append(dict(id=key,group=group,views=[row]));groups[group]=groups.get(group,0)+1
        if len(selected)==256:break
    if len(selected)!=256:raise ValueError('Prespecified TRAIN diagnostic sample unavailable')
    out=root/IDENTITY;out.mkdir();PUBLIC.mkdir(parents=True,exist_ok=False);write_state(out/'panel.json',selected);cfg=ParcaeCandidateConfig(**old['model']);noise=[]
    for row in selected:
        prefix=row['views'][0]['prefix_ids'];seed=int(hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest()[:16],16)%(2**63);noise.append(initial_state(cfg,0,seed)[:,:128].numpy()[0])
    np.save(out/'noise.npy',np.stack(noise));keys=['one-bridge','recurrent-bridge','one-bridge-work'];extras=['scripts/run_natural_chain_train_diagnostic.py','scripts/read_natural_chain_train_native.py','scripts/audit_natural_chain_train_cpu.py','compute_trainium/readout_panel_scope.py','docs/NATURAL_CHAIN_TRAIN_DIAGNOSTIC_20261025.md']
    sources={**old['source_sha256'],**{p:file_sha256(p) for p in extras}};worker,host=split_snapshot_sources(sources);plan=copy.deepcopy(old)
    plan.update(identity=IDENTITY,registered_unix=time.time(),panel_role='train_only_post_pilot_diagnostic',cases={k:old['cases'][k] for k in keys},
        panel=str((out/'panel.json').relative_to(root)),panel_sha256=file_sha256(out/'panel.json'),noise=str((out/'noise.npy').relative_to(root)),noise_sha256=file_sha256(out/'noise.npy'),
        source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,seconds_limit=1800,phase_seconds_limit=600,cpu_phase_seconds_limit=600,
        original_pilot_report_sha256=file_sha256(prior/'report.json'),train_views_sha256=file_sha256(source),selected_train_tasks=256,selected_source_groups=len(groups),
        root_comm_ports=reserve_ports(root,[IDENTITY+'-'+k for k in keys]),scope=Path(extras[-1]).read_text())
    write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    original.IDENTITY=IDENTITY;original.PUBLIC=PUBLIC;original.Execution=ScopedExecution;original.run(root,plan)
    file=out/'report.json';report=json.loads(file.read_text());report.update(status='COMPLETE_POST_PILOT_TRAIN_DIAGNOSTIC',panel_role=plan['panel_role'],heldout_result=False);write_state(file,report);write_state(PUBLIC/'report.json',report);write_state(PUBLIC/'status.json',dict(status=report['status'],assigned_chips=0))


if __name__=='__main__':main()
