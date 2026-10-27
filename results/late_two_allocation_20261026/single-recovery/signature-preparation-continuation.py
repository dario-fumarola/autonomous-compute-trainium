"""Finish the exact registered CPU signature migration after scoped directory permission repair."""
from pathlib import Path
import json,time
import torch
from scripts.run_late_two_single_recovery import IDENTITY,PARENT,KEY,PUBLIC,read,checkpoint,verify_readback
from compute_trainium.future_span_scientific_state import signature
from compute_trainium.late_allocation_recovery import rebind_same_endpoint
from compute_trainium.training import file_sha256,atomic_checkpoint
from compute_trainium.queue import write_state
from scripts.qualify_native import digest
root=Path('/run/user/1000/autonomous-compute/instance-nvme');out=root/IDENTITY;file=out/'plan.json'
plan=read(file);old_file=root/PARENT/'plan.json';old=read(old_file);old_sha=file_sha256(old_file)
assert plan['original_parent_plan_sha256']==old_sha
assert all(file_sha256(p)==sha for p,sha in plan['source_sha256'].items())
assert not (out/'migration.json').exists() and not (out/'execution').exists()
phase=PARENT+'-'+KEY+'-one-finish';row=read(root/'runs'/phase/'report.json');capture=(phase,row)
verify_readback(root,old_file,KEY,capture);point=checkpoint(capture)
original=root/'runs'/phase/'checkpoint-u3300.pt';saved=torch.load(original,map_location='cpu',mmap=True,weights_only=False)
new_sha=file_sha256(file)
rebound=rebind_same_endpoint(saved,signature(old_sha,old,old['cases'][KEY]),signature(new_sha,plan,plan['cases'][KEY]),old['cases'][KEY],plan['cases'][KEY],case_id=KEY,update=3300)
before=digest({k:v for k,v in saved.items() if k!='signature'})
anchor_name=IDENTITY+'-signature-only-anchor';migrated=root/'runs'/anchor_name/'checkpoint-u3300.pt'
assert not migrated.exists();atomic_checkpoint(migrated,rebound)
check=torch.load(migrated,map_location='cpu',mmap=True,weights_only=False)
assert digest({k:v for k,v in check.items() if k!='signature'})==before and check['signature']==rebound['signature']
derived=dict(point,checkpoint_sha256=file_sha256(migrated));derived_capture=dict(update=3300,checkpoint_records=[derived])
verify_readback(root,file,KEY,(anchor_name,derived_capture))
migration=dict(status='PASS_SIGNATURE_ONLY_SAME_ENDPOINT_RECOVERY',original_checkpoint_sha256=point['checkpoint_sha256'],new_checkpoint_sha256=derived['checkpoint_sha256'],original_plan_sha256=old_sha,new_plan_sha256=new_sha,all_other_state_exact=True,inherited_replica_agreement=True,anchor_directory=anchor_name,derived_capture=derived_capture,preparation_continuation_script_sha256=file_sha256(__file__),preparation_failed_log_sha256=file_sha256(root/'late-single-recovery-preparation.log'),optimizer_updates=0,assigned_chips=0,scope='CPU preparation resumes only after creating its exact registered anchor directory with host ownership. All original state except signature.plan_sha256 is verified unchanged. Derived metadata is not a Native phase report.')
write_state(out/'migration.json',migration);write_state(PUBLIC/'migration.json',migration);write_state(PUBLIC/'plan.json',plan)
write_state(PUBLIC/'status.json',dict(status='PREPARED_SINGLE_RECOVERY_NOT_ADMITTED',assigned_chips=0))
print(json.dumps(dict(status=migration['status'],all_other_state_exact=True,plan_sha256=new_sha)))
