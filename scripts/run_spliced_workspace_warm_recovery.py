"""Finite sparse whole-training-path qualification and full-global replay."""
import argparse,json,os,shutil,time
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from scripts.run_scale_sharding_fixture import AllChipExecution
ID='spliced-workspace-warm-v2';PUBLIC=Path('results/spliced_workspace_warm_20261028/recovery-v2')


def prepare(root):
 if managed_running():raise RuntimeError('Exclusive closed prior owner required')
 parent=root/'spliced-workspace-native-v1';r=json.loads((parent/'report.json').read_text());p=json.loads((parent/'plan.json').read_text())
 if r['status']!='PASS_SPARSE_NATIVE_GRADIENT_UPDATE_RECOVERY':raise AssertionError('Full Native update/moment gate required')
 if shutil.disk_usage(root).free<61*1024**3:raise RuntimeError('All64 full state+gradient+compiler allowance and35GiB reserve required')
 sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs'] for f in Path(d).rglob('*.py') if f.is_file()};sources['scripts/run_native_isolated.sh']=file_sha256('scripts/run_native_isolated.sh')
 worker,host=split_snapshot_sources(sources);projection=project_snapshot_inodes(Path('.'),2)
 if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('Both warm/replay snapshots and artifact inode reserve required')
 from compute_trainium.workspace_view_batches import WorkspaceViewBatches
 data=WorkspaceViewBatches(root/'workspace-view-batches-v1','train');meta=[json.loads(v) for v in (root/'workspace-source-views-v2/train-views.jsonl').read_text().splitlines()]
 groups={meta[int(data.order[k])]['group'] for k in range(6464)};audit=next(k for k in range(6464,len(data)) if meta[int(data.order[k])]['group'] not in groups)
 report=root/'workspace-view-batches-v1/report.json';data_receipt=json.loads(report.read_text())
 if any(file_sha256(root/'workspace-view-batches-v1'/n)!=h for n,h in data_receipt['records']['train']['files'].items()):raise AssertionError('Frozen TRAIN arrays changed')
 original_path=root/'spliced-workspace-warm-v1/plan.json';original=json.loads(original_path.read_text());remaining=original['absolute_deadline_unix']-time.time()
 if remaining<1200:raise RuntimeError('Complete recovery allowance must fit original deadline')
 phases=[ID+'-warm',ID+'-replay'];start=time.time();plan=dict(identity=ID,purpose='spliced_workspace_warm_and_replay_qualification',world=64,worker_memory_gib=192,registered_unix=start,seconds_limit=int(remaining),phase_seconds_limit=min(2400,int(remaining)),absolute_deadline_unix=original['absolute_deadline_unix'],clock_admission_waiver=True,
 source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,root_comm_ports=reserve_ports(root,phases),phases={n:dict(mode=m) for n,m in zip(phases,['warm','replay'])},inode_projection=projection,
 model_config=p['model_config'],checkpoint_relative=p['checkpoint_relative'],checkpoint_sha256=p['checkpoint_sha256'],prior_native_report_sha256=file_sha256(parent/'report.json'),data_relative='workspace-view-batches-v1',data_report_sha256=file_sha256(report),mature_audit_index=audit,qualification_source_groups=len(groups),warm_phase=phases[0],cold_updates=1,warm_updates=100,actual_qualification_optimizer_updates=101,scientific_optimizer_updates=0,anchors=list(range(31,159,4)),mature_anchors=list(range(31,159,4)),query_count=32,workspace_mode='latent',sample_seed_base=9821,retained_candidate_prefix=True,mode='latent',mix=.5,lr=1e-5,global_batch=64,early_warm_stability_screen=dict(at_update=6,mean_last5_seconds_limit=8),
 initial_reference_relative=p['gradient_relative'],initial_reference_sha256=p['gradient_sha256'],parent_failed_plan_sha256=file_sha256(original_path),native_cache_environment=dict(RESEARCH_HLO_EAGER_CACHE='0',RESEARCH_HLO_COMPILE_CACHE='0',RESEARCH_NKI_CACHE_URL='/work/instance-nvme/runtime/nki-intermediate-cache'),
 scope='Same101 updates,C0,RNG,source order and original absolute deadline.Cache repair bypasses persistent HLO IR with stale container-local NKI binary paths and places new NKI binaries on persistent storage.Cold full179-gradient check precedes all fitting;2% and fail-closed compiler checks unchanged.Prospectively retained candidate prefix,fresh C0/fresh AdamW.One cold+100warm UNIFORM spliced NTP updates,32 RNG-only selected positions per view.Each authentic label exactly once.All101 prefix updates/source/costs must count toward any later discovery,not an extra hidden warm-up.No TUNE or duration selection from these results.Admission of a larger continuation remains conditional on full gates and a new measured budget.Full101 counter/state retained.Original one-step qualification updates excluded.All64 full actor/AdamW/RNG states,unclipped mature gradient on a disjoint TRAIN group;fresh-process full-global hash/RNG restore and exact zero-update gradient replay.Independent mature HF audit still required before scientific fit.No TUNE/DEV read,no learned allocation or quality claim;512GiB/35GiB/2% gates unchanged.')
 out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True)
 for d in [out,PUBLIC]:write_state(d/'plan.json',plan)


def run(root):
 out=root/ID;e=AllChipExecution(root,out/'plan.json',out/'execution');p=e.plan
 os.environ.update(p['native_cache_environment'])
 if file_sha256(root/p['initial_reference_relative'])!=p['initial_reference_sha256']:raise AssertionError('Frozen cold reference changed')
 r=dict(status='STOPPED_SPARSE_WARM_QUALIFICATION',scientific_optimizer_updates=0,records={})
 try:
  for mode,status in [('warm','COMPLETE_SPARSE_WARM_PENDING_MATURE_AUDIT'),('replay','PASS_SPARSE_FULL_GLOBAL_RESTORE_AND_GRADIENT_REPLAY')]:
   phase=ID+'-'+mode;args=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+ID+'/plan.json','--plan-sha',e.sha,'--comm-port',str(p['root_comm_ports'][phase]),'--mode',mode]
   r['records'][mode]=e.native(phase,'scripts.qualify_sparse_workspace_warm',args,status)
   for d in [out,PUBLIC]:write_state(d/'progress.json',r)
  r['status']='COMPLETE_SPARSE_WARM_AND_REPLAY_PENDING_MATURE_HF_AUDIT'
 except BaseException as error:r.update(error_type=type(error).__name__,error=str(error));raise
 finally:
  path=root/'runs'/p['warm_phase']/'updates.json';r['observed_completed_qualification_updates']=len(json.loads(path.read_text())) if path.exists() else 0;r['interrupted_update_may_be_unrecorded']=r['status'].startswith('STOPPED')
  e.close(out/'execution-costs.json');r['plan_sha256']=e.sha
  for d in [out,PUBLIC]:write_state(d/'report.json',r);write_state(d/'execution-costs.json',json.loads((out/'execution-costs.json').read_text()))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args();run(a.root) if a.run else prepare(a.root)
