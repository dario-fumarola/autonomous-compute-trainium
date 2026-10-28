"""Freeze the source-volume question before its six endpoint fits."""
import argparse,json,time,os,shutil
from pathlib import Path


def run(root):
 from compute_trainium.training import file_sha256
 from compute_trainium.workspace_view_batches import WorkspaceViewBatches
 from compute_trainium.execution_isolation import managed_running
 from compute_trainium.native_comm_ports import reserve_ports
 from compute_trainium.native_inode_projection import project_snapshot_inodes
 from compute_trainium.native_snapshot_manifest import split_snapshot_sources
 start=time.time()
 if managed_running() or os.environ.get('USE_TORCH_XLA')!='0':raise RuntimeError('Closed Native owner required')
 if shutil.disk_usage(root).free<44*1024**3:raise RuntimeError('9GiB artifacts plus35GiB reserve required')
 available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
 if available<(512+192)*1024**3:raise RuntimeError('Protected reserve plus Native worker allowance required')
 previous=root/'residual-workspace-screen-v1';old=json.loads((previous/'plan.json').read_text())
 if json.loads((previous/'report.json').read_text())['status']!='COMPLETE_FROZEN_RESIDUAL_SCREEN':raise AssertionError('Completed original negative required')
 if file_sha256(root/old['checkpoint_relative'])!=old['checkpoint_sha256']:raise AssertionError('Frozen C0 changed')
 arrays=root/old['data_relative'];receipt=json.loads((arrays/'report.json').read_text())
 for split in ['train','tune']:
  for name,sha in receipt['records'][split]['files'].items():
   if file_sha256(arrays/name)!=sha:raise AssertionError('Authentic source array changed')
 data=WorkspaceViewBatches(arrays,'train');selected=set(int(x) for x in data.order[:8192]);groups=[]
 for row,line in enumerate((root/'workspace-source-views-v2/train-views.jsonl').open()):
  if row in selected:groups.append(json.loads(line)['group'])
 if len(groups)!=8192 or set(groups)&set(old['evaluation_groups']):raise AssertionError('Exact source-separated TRAIN views required')
 caches={}
 for split in ['train','tune']:
  for rank in range(64):
   path=root/'runs'/old['feature_phase']/f'PRIVATE-rank{rank}-{split}.pt';caches[str(path.relative_to(root))]=file_sha256(path)
 identity='residual-data-volume-v1';phase=identity+'-features';out=root/identity;public=Path('results/residual_data_volume_20261028');out.mkdir(mode=0o700);public.mkdir()
 modes={s+'_'+m:m for s in ['fresh','repeat'] for m in ['linear','one','two']};resumes={}
 for arm,mode in modes.items():
  path=previous/f'PRIVATE-{mode}-state.pt';resumes[arm]=dict(relative=str(path.relative_to(root)),sha256=file_sha256(path),updates=512)
 sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs'] for f in Path(d).rglob('*.py')};sources['scripts/run_native_isolated.sh']=file_sha256('scripts/run_native_isolated.sh');worker,host=split_snapshot_sources(sources);projection=project_snapshot_inodes(Path('.'),1)
 if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Protected snapshot inode reserve')
 keep=['model_config','checkpoint_relative','checkpoint_sha256','data_relative','evaluation_anchors','evaluation_groups','evaluation_split','weights','native_cache_environment','cpu_references','seed','lr','weight_decay','width','queries','batch_views']
 p={key:old[key] for key in keep};p.update(identity=identity,purpose='source_diversity_vs_repeated_updates',volume_comparison=True,registered_unix=start,absolute_deadline_unix=start+2700,seconds_limit=2700,phase_seconds_limit=600,cpu_phase_seconds_limit=1800,clock_admission_waiver=True,world=64,worker_memory_gib=192,source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,inode_projection=projection,root_comm_ports=reserve_ports(root,[phase]),phases={phase:dict(mode='frozen-features')},train_views=8192,arm_modes=list(modes),branch_modes=modes,arm_unique_views={arm:8192 if arm.startswith('fresh') else 2048 for arm in modes},updates=2048,resume_states=resumes,backbone_frozen=True,head_frozen=True,feature_phase=phase,feature_segments=dict(train=[old['feature_phase'],phase],tune=[old['feature_phase']]),feature_extraction=[dict(split='train',start=2048,count=6144)],retained_feature_sha256=caches,source_policy_sha256=file_sha256('docs/RESIDUAL_DATA_VOLUME_20261028.md'),scope=Path('docs/RESIDUAL_DATA_VOLUME_20261028.md').read_text(),training_source_groups=len(set(groups)),train_tune_source_group_overlap=0,training_domain_counts=[sum(data.at(i)['domain']==d for i in range(8192)) for d in range(3)],source_array_receipt_sha256=file_sha256(arrays/'report.json'),retained_prefix_plan_sha256=file_sha256(previous/'plan.json'),retained_prefix_cost_policy='All512 original updates per mode and old feature extraction count.Logical paths share retained prefix;actual cost counts it once.',retained_prefix_costs=json.loads((Path('results/residual_workspace_screen_20261028')/'complete-costs.json').read_text()))
 for folder in [out,public]:(folder/'plan.json').write_text(json.dumps(p,indent=2)+'\n')
 report=dict(status='PASS_RESIDUAL_VOLUME_ADMISSION',source_group_count=len(set(groups)),overlap=0,retained_cache_files=len(caches),free_bytes=shutil.disk_usage(root).free,free_inodes=os.statvfs(root).f_favail,wall_seconds=time.time()-start,assigned_chips=0)
 for folder in [out,public]:(folder/'admission.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
