"""Finite saved-writer/reader readout,not another fit."""
import argparse,json,os,shutil,time
from pathlib import Path
from compute_trainium.execution_isolation import managed_running
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_inode_projection import project_snapshot_inodes
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.training import file_sha256
from compute_trainium.queue import write_state
from scripts.resume_workspace_discovery_evaluation import validate_endpoints
from scripts.run_scale_sharding_fixture import AllChipExecution
ID='workspace-crossover-v1';PUBLIC=Path('results/workspace_crossover_20261028')


def prepare(root):
 if managed_running():raise RuntimeError('Closed readout required')
 parent=root/'workspace-readout-diagnostic-v2';r=json.loads((parent/'report.json').read_text())
 if r['status']!='COMPLETE_WORKSPACE_READOUT_DIAGNOSTIC':raise AssertionError('Finish all registered readouts first')
 p=json.loads((parent/'plan.json').read_text());proof=validate_endpoints(root,p)
 if proof!=p['retained_endpoint_proofs']:raise AssertionError('Endpoint changed')
 sources={str(f):file_sha256(f) for d in ['compute_trainium','scripts','configs'] for f in Path(d).rglob('*.py') if f.is_file()}
 sources['scripts/run_native_isolated.sh']=file_sha256('scripts/run_native_isolated.sh');worker,host=split_snapshot_sources(sources)
 projection=project_snapshot_inodes(Path('.'),1)
 if os.statvfs(root).f_favail<projection['required_free_inodes'] or shutil.disk_usage(root).free<36*1024**3:raise RuntimeError('Snapshot/artifact reserve required')
 phase=ID+'-evaluation';notes=p['identity']+'-evaluation'
 p.update(identity=ID,purpose='workspace_writer_reader_crossover',registered_unix=time.time(),seconds_limit=1200,phase_seconds_limit=1200,worker_memory_gib=128,absolute_deadline_unix=time.time()+1200,
 source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,root_comm_ports=reserve_ports(root,[phase]),phases={phase:dict(mode='evaluation')},inode_projection=projection,
 saved_note_run=notes,saved_note_sha256={str(k):file_sha256(root/'runs'/notes/f'rank{k}-PRIVATE-generated-notes.pt') for k in range(64)},scientific_optimizer_updates=0,
 scope='Only C0/workspace writers crossed with C0/workspace readers.Same320 prior TUNE anchors,saved G1 samples and prepend layout.No refitting,oracle or prompt search.New finite20-minute zero-update follow-up;all prior fit/readout deadlines and receipts remain unchanged.128GiB admission follows measured88GiB five-model readout;512GiB host/35GiB disk/2% numerical limits unchanged.')
 out=root/ID;out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',p);write_state(PUBLIC/'plan.json',p)


def run(root):
 out=root/ID;e=AllChipExecution(root,out/'plan.json',out/'execution');p=e.plan;r=dict(status='STOPPED_WORKSPACE_CROSSOVER',optimizer_updates=0)
 try:
  if validate_endpoints(root,p)!=p['retained_endpoint_proofs']:raise AssertionError('Endpoint proofs changed')
  phase=ID+'-evaluation';args=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+ID+'/plan.json','--plan-sha',e.sha,'--comm-port',str(p['root_comm_ports'][phase])]
  r['evaluation']=e.native(phase,'scripts.evaluate_workspace_crossover',args,'COMPLETE_WORKSPACE_WRITER_READER_CROSSOVER');r['status']='COMPLETE_WORKSPACE_WRITER_READER_CROSSOVER'
 except BaseException as error:r.update(error_type=type(error).__name__,error=str(error));raise
 finally:
  e.close(out/'execution-costs.json');r['plan_sha256']=e.sha
  for d in [out,PUBLIC]:write_state(d/'report.json',r);write_state(d/'execution-costs.json',json.loads((out/'execution-costs.json').read_text()))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args();run(a.root) if a.run else prepare(a.root)
