"""Finite zero-update calibration diagnostic on retained scientific models."""
import argparse,json,os,time,shutil
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from scripts.run_scale_sharding_fixture import AllChipExecution
ID='spliced-workspace-readout-v1';PUBLIC=Path('results/spliced_workspace_readout_20261028')


def prepare(root,*,source_reserve=False):
 if managed_running():raise RuntimeError('All prior scientific workers must be closed')
 parent=root/'spliced-workspace-discovery-v1';p=json.loads((parent/'plan.json').read_text());r=json.loads((parent/'report.json').read_text())
 if r['status']!='COMPLETE_SPLICED_WORKSPACE_DISCOVERY' or any(a['status']!='PASS_SPLICED_INDEPENDENT_MATURE_CPU_AUDIT' for a in r['audits'].values()):raise AssertionError('Complete uniformly trained and audited endpoints required')
 if shutil.disk_usage(root).free<36*1024**3:raise RuntimeError('OneGiB diagnostic allowance plus35GiB reserve required')
 sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs'] for f in Path(d).rglob('*.py') if f.is_file()};sources['scripts/run_native_isolated.sh']=file_sha256('scripts/run_native_isolated.sh');worker,host=split_snapshot_sources(sources);projection=project_snapshot_inodes(Path('.'),1)
 if os.statvfs(root).f_favail<projection['required_free_inodes']:raise RuntimeError('One complete snapshot and inode reserve required')
 checkpoint_hashes={}
 for arm in ['latent','constant','ordinary']:
  audit=json.loads((root/('spliced-workspace-discovery-audit-'+arm+'-v1')/'plan.json').read_text());actual=[file_sha256(root/'runs'/p['arm_runs'][arm]/f'rank{k}-u{p["updates"][arm]}.zip') for k in range(64)]
  if actual!=audit['checkpoint_sha256']:raise AssertionError('Prior independently audited scientific state changed')
  checkpoint_hashes[arm]=actual
 start=time.time();phase=ID+'-evaluation';reserve_fields={}
 if source_reserve:
  from compute_trainium.workspace_view_batches import WorkspaceViewBatches
  from compute_trainium.spliced_workspace_readout import source_reserve_indices
  data=WorkspaceViewBatches(root/p['data_relative'],'train');meta=[json.loads(v) for v in (root/'workspace-source-views-v2/train-views.jsonl').read_text().splitlines()];groups=[meta[int(data.order[i])]['group'] for i in range(len(data))];trained=max(p['updates'].values())*64
  indices=source_reserve_indices(groups,trained_views=trained,count=192,excluded_groups=[groups[p['mature_audit_index']]])
  original=json.loads((root/'spliced-workspace-readout-v1/plan.json').read_text())
  if original['absolute_deadline_unix']-start<300:raise RuntimeError('Complete follow-up must fit original diagnostic deadline')
  reserve_fields=dict(calibration_views=192,calibration_indices=indices,calibration_source_groups=len({groups[i] for i in indices}),trained_source_views_excluded=trained,calibration_groups_disjoint_from_all_backbone_training=True,parent_readout_plan_sha256=file_sha256(root/'spliced-workspace-readout-v1/plan.json'),original_diagnostic_deadline_unix=original['absolute_deadline_unix'])
 plan=dict(identity=ID,purpose='spliced_workspace_global_train_readout',registered_unix=start,world=64,worker_memory_gib=192,seconds_limit=1800,phase_seconds_limit=1200,cpu_phase_seconds_limit=120,absolute_deadline_unix=start+1800,clock_admission_waiver=True,source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,root_comm_ports=reserve_ports(root,[phase]),phases={phase:dict(mode='zero-update-readout')},inode_projection=projection,
 model_config=p['model_config'],checkpoint_relative=p['checkpoint_relative'],checkpoint_sha256=p['checkpoint_sha256'],checkpoint_hashes=checkpoint_hashes,arm_runs=p['arm_runs'],updates=p['updates'],data_relative=p['data_relative'],data_report_sha256=p['data_report_sha256'],pause_token_id=p['pause_token_id'],native_cache_environment=p['native_cache_environment'],prior_report_sha256=file_sha256(parent/'report.json'),prior_scores=r['analysis']['independent_scores'],calibration_views=640,evaluation_anchors=448,evaluation_groups=p['evaluation_groups'],evaluation_split='tune',alphas=[0,.05,.1,.2,.35,.5],temperatures=[1.,.9,1.1],optimizer_updates=0,
 primary='Held-out own-workspace selected utility and full32 NLL vs equally TRAIN-temperature-calibrated ordinary.Ordinary alpha fixed0;workspace conditions use one TRAIN-chosen global alpha/temperature.',
 stopping_rule='One finite readout diagnostic,no new backbone fit.If TRAIN chooses0 or held-out workspace utility remains nonpositive,close this fixed-reader recipe;no larger run or router automatically admitted.A nonzero chosen mixer and positive utility is development information only,not proof of efficient reasoning.',
 scope='Fixed pretrained models.Each workspace readout linearly interpolates raw logits with the existing .5-mixed logits;alpha stays0.. .5.No new latent steps or parameters learned in the backbone.Choose global settings from640 TRAIN views only,then freeze before448 reused TUNE views.No per-token target-aware selection,no held-out grid optimization.All source text and private per-view rows remain private.Same original raw/mixed endpoints must reproduce within1e-5,all parameter versions unchanged,no fallback.18-point joint grid vs3 temperatures for raw;calibration/evaluation cost reported separately.No new from-scratch,allocation or confirmation claim.')
 if source_reserve:
  plan.update(reserve_fields);plan['absolute_deadline_unix']=reserve_fields['original_diagnostic_deadline_unix'];plan['seconds_limit']=int(plan['absolute_deadline_unix']-start);plan['phase_seconds_limit']=min(1200,plan['seconds_limit']);plan['scope']+=' Source-reserve correction:192 deterministic unused TRAIN views from groups disjoint from all60160 backbone source views and the mature audit group.No grid change or new duration;original30-minute absolute deadline retained.First in-sample calibration result/cost preserved.'
 out=root/ID;out.mkdir();PUBLIC.mkdir()
 for d in [out,PUBLIC]:write_state(d/'plan.json',plan)


def run(root):
 out=root/ID;e=AllChipExecution(root,out/'plan.json',out/'execution');p=e.plan;os.environ.update(p['native_cache_environment']);r=dict(status='STOPPED_SPLICED_GLOBAL_TRAIN_READOUT',optimizer_updates=0)
 try:
  phase=ID+'-evaluation';r['evaluation']=e.native(phase,'scripts.evaluate_spliced_workspace_readout',['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+ID+'/plan.json','--plan-sha',e.sha,'--comm-port',str(p['root_comm_ports'][phase])],'COMPLETE_SPLICED_GLOBAL_TRAIN_READOUT')
  from compute_trainium.workspace_discovery_analysis import clustered_interval
  rows={arm:sorted([v for k in range(64) for v in json.loads((root/'runs'/phase/f'rank{k}-{arm}-tune-scores.json').read_text())],key=lambda x:x['index']) for arm in ['c0','ordinary','latent','constant']}
  for arm,values in rows.items():
   if [v['index'] for v in values]!=list(range(448)):raise AssertionError('Exact fixed held-out panel coverage required')
  domains=[v['domain'] for v in rows['ordinary']];contrasts={}
  for name,left,lf,right,rf in [('ordinary_minus_latent_calibrated_policy32','ordinary','policy32_nll','latent','policy32_nll'),('latent_absolute_anchor_utility_same_temperature','latent','same_temperature_raw_anchor_nll','latent','workspace_anchor_nll'),('latent_calibrated_raw_minus_calibrated_policy32','latent','calibrated_raw32_nll','latent','policy32_nll'),('constant_minus_latent_calibrated_policy32','constant','policy32_nll','latent','policy32_nll')]:
   contrasts[name]=clustered_interval([a[lf]-b[rf] for a,b in zip(rows[left],rows[right],strict=True)],p['evaluation_groups'],domains)
  r.update(status='COMPLETE_SPLICED_GLOBAL_TRAIN_READOUT',contrasts=contrasts,confirmation=False,terminal_alpha_spent=0)
 except BaseException as error:r.update(error_type=type(error).__name__,error=str(error));raise
 finally:
  e.close(out/'execution-costs.json');r['plan_sha256']=e.sha
  for d in [out,PUBLIC]:write_state(d/'report.json',r);write_state(d/'execution-costs.json',json.loads((out/'execution-costs.json').read_text()))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');p.add_argument('--source-reserve',action='store_true');a=p.parse_args()
 if a.source_reserve:ID='spliced-workspace-readout-v2';PUBLIC=Path('results/spliced_workspace_readout_20261028/source-reserve-v2')
 run(a.root) if a.run else prepare(a.root,source_reserve=a.source_reserve)
