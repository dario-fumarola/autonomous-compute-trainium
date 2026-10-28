"""Register a finite learnability screen and independent frozen-base oracle."""
import argparse,json,time,os,shutil
from pathlib import Path


def run(root):
 import torch
 from transformers import AutoModelForCausalLM
 from compute_trainium.training import file_sha256
 from compute_trainium.workspace_view_batches import WorkspaceViewBatches
 from compute_trainium.sparse_workspace_training import pack_view
 from compute_trainium.execution_isolation import managed_running
 from compute_trainium.native_comm_ports import reserve_ports
 from compute_trainium.native_inode_projection import project_snapshot_inodes
 from compute_trainium.native_snapshot_manifest import split_snapshot_sources
 if managed_running() or os.environ.get('USE_TORCH_XLA')!='0':raise RuntimeError('Closed owner,CPU reference required')
 if shutil.disk_usage(root).free<40*1024**3:raise RuntimeError('5GiB artifact allowance plus35GiB reserve required')
 parent=json.loads((root/'spliced-workspace-discovery-v1/plan.json').read_text());identity='residual-workspace-screen-v1';out=root/identity;public=Path('results/residual_workspace_screen_20261028');out.mkdir(mode=0o700);public.mkdir();started=time.time();torch.set_num_threads(12)
 model=AutoModelForCausalLM.from_pretrained(root/'workspace-olmo2-stage1-base-v1',local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').eval().requires_grad_(False)
 if file_sha256(root/parent['checkpoint_relative'])!=parent['checkpoint_sha256']:raise AssertionError('C0 changed')
 state=torch.load(root/parent['checkpoint_relative'],weights_only=True,mmap=True)['model'];model.load_state_dict(state);del state;data=WorkspaceViewBatches(root/parent['data_relative'],'train');indices=[next(i for i in range(64) if data.at(i)['domain']==domain) for domain in range(3)];refs=[]
 with torch.no_grad():
  for index in indices:
   tokens,_=pack_view(data.at(index),model.config.pad_token_id);h=model.model(tokens,use_cache=False).last_hidden_state;f=out/f'PRIVATE-reference-{index}.pt';torch.save(dict(features=h,logits=model.lm_head(h[:,127:159])),f);refs.append(dict(index=index,relative=str(f.relative_to(root)),sha256=file_sha256(f)))
 del model
 selected=set(int(x) for x in data.order[:2048]);groups=[]
 for row,line in enumerate((root/'workspace-source-views-v2/train-views.jsonl').open()):
  if row in selected:groups.append(json.loads(line)['group'])
 if len(groups)!=2048 or set(groups)&set(parent['evaluation_groups']):raise AssertionError('Source-separated training and evaluation required')
 phase=identity+'-features';sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs'] for f in Path(d).rglob('*.py')};sources['scripts/run_native_isolated.sh']=file_sha256('scripts/run_native_isolated.sh');worker,host=split_snapshot_sources(sources);projection=project_snapshot_inodes(Path('.'),1)
 if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Protected snapshot inode reserve')
 p=dict(parent,identity=identity,purpose='frozen_base_residual_learnability',registered_unix=started,absolute_deadline_unix=started+2700,seconds_limit=2700,phase_seconds_limit=600,cpu_phase_seconds_limit=1800,clock_admission_waiver=True,world=64,worker_memory_gib=192,source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,inode_projection=projection,root_comm_ports=reserve_ports(root,[phase]),phases={phase:dict(mode='frozen-features')},cpu_references=refs,train_views=2048,evaluation_anchors=448,arm_modes=['linear','one','two'],updates=512,batch_views=4,queries=32,seed=7285,lr=.0003,weight_decay=.1,width=128,backbone_frozen=True,head_frozen=True,feature_phase=phase,source_policy_sha256=file_sha256('docs/FROZEN_RESIDUAL_SCREEN_20261028.md'),scope=Path('docs/FROZEN_RESIDUAL_SCREEN_20261028.md').read_text())
 p['training_source_groups']=len(set(groups));p['train_tune_source_group_overlap']=0;p['training_domain_counts']=[sum(data.at(i)['domain']==d for i in range(2048)) for d in range(3)]
 p['source_array_receipt_sha256']=file_sha256(root/parent['data_relative']/'report.json')
 # Avoid inheriting old arm budgets/claims;only source/model provenance remains.
 for key in ['arm_runs','prefixes','nominal_operations_per_view','work_ratio']:p.pop(key,None)
 for folder in [out,public]:(folder/'plan.json').write_text(json.dumps(p,indent=2)+'\n')
 report=dict(status='PASS_FROZEN_RESIDUAL_CPU_REFERENCE',wall_seconds=time.time()-started,optimizer_updates=0,assigned_chips=0,reference_count=len(refs))
 for folder in [out,public]:(folder/'cpu-reference.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
