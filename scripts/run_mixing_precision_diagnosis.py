"""Bounded no-update localization after the mature cheap-control gate failed."""
import argparse
import concurrent.futures as cf
import json
from pathlib import Path
import shutil
import signal
import sys
import time

import numpy as np
import torch
from compute_trainium.execution_isolation import managed_running
from compute_trainium.future_span import FutureSpanConfig
from compute_trainium.future_span_execution import Execution
from compute_trainium.future_span_training import FutureSpanObjective,objective_coefficients,training_window
from compute_trainium.inode_reserve import require_inode_reserve
from compute_trainium.mixing_precision_probe import PrecisionMixingProbe
from compute_trainium.native_comm_ports import reserve_ports
from compute_trainium.native_snapshot_manifest import split_snapshot_sources
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_mixing_control import MixingOnlyFutureSpanCandidate
from compute_trainium.parcae_training_candidate import gradient_comparison
from compute_trainium.queue import write_state
from compute_trainium.training import PackedCorpus,file_sha256
from scripts.qualify_native import digest

IDENTITY='mixing-precision-diagnosis-v1'
PUBLIC=Path('results/mixing_precision_diagnosis_20261026')
DOC='docs/MIXING_PRECISION_DIAGNOSIS_20261026.md'


def read(p):return json.loads(p.read_text())


def guards(root):
    require_inode_reserve([root,Path('/')])
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if available<544*1024**3 or min(shutil.disk_usage(root).free,shutil.disk_usage('/').free)<35*1024**3:
        raise RuntimeError('Protected diagnostic reserves failed')


def prepare(root):
    out=root/IDENTITY
    if (out/'plan.json').exists():return read(out/'plan.json')
    started=time.time();guards(root)
    parent_dir=root/'fixed-control-replication-main-v1';terminal=read(parent_dir/'report.json')
    failed=parent_dir/'gradients/cheap-s5231-work/report.json';failure=read(failed)
    if terminal['status']!='STOPPED_FIXED_CONTROL_DEVELOPMENT_REPLICATION' or failure['status']!='FAIL_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT':
        raise ValueError('Preserved actual final-control numerical failure required')
    parent_file=parent_dir/'mixing-plan.json';parent=read(parent_file);case=parent['cases']['cheap-s5231-work']
    folder=root/'runs/fixed-control-replication-main-v1-cheap-s5231-work-finish';native=read(folder/'report.json')
    checkpoint=folder/'checkpoint-u4594.pt';record=next(v for v in native['checkpoint_records'] if v['update']==4594)
    if (failure['checkpoint_sha256']!=record['checkpoint_sha256'] or file_sha256(checkpoint)!=record['checkpoint_sha256']
            or not record['all_complete_replica_states_equal']):raise AssertionError('Exact failed trained state required')
    capture=root/'runs/fixed-control-replication-main-v1-cheap-s5231-work-gradient';captured=read(capture/'report.json')
    extras=['scripts/run_mixing_precision_diagnosis.py','scripts/capture_mixing_precision.py',
            'scripts/compare_mixing_precision_cpu.py','compute_trainium/mixing_precision_probe.py',DOC]
    sources={**parent['source_sha256'],**{f:file_sha256(f) for f in extras}}
    if any(file_sha256(f)!=sha for f,sha in sources.items()):raise AssertionError('Preserved original/diagnostic source changed')
    point=dict(parent_plan=str(parent_file.relative_to(root)),parent_plan_sha256=file_sha256(parent_file),case='cheap-s5231-work',
        checkpoint=str(checkpoint.relative_to(root)),checkpoint_sha256=record['checkpoint_sha256'],
        failed_capture=str(capture.relative_to(root)),failed_report_sha256=file_sha256(failed))
    # Before native admission,all branches with autocast disabled must preserve
    # the original FP32 local function on an actual next-TRAIN window.
    torch.set_num_threads(12);torch.set_num_interop_threads(1)
    saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    c=ParcaeCandidateConfig(**parent['model']);span=FutureSpanConfig(**parent['future_config'])
    packed=PackedCorpus.from_mapped(root/parent['data']['packed'],1024,23,parent['data']['corpus_sha256'])
    row=captured['rows'][0];values=list(training_window(packed,row['window'],c,span,case['state_seed']))
    noise_file=capture/f"h0-rank{row['rank']}.npy"
    if file_sha256(noise_file)!=captured['noise_sha256_by_rank'][row['rank']]:raise AssertionError('Original supplied state bytes changed')
    supplied=np.load(noise_file,mmap_mode='r')
    values[3]=torch.from_numpy(np.array(supplied[row['micro']:row['micro']+1],copy=True))
    if digest(values[3])!=row['h0_digest']:raise AssertionError('Original supplied latent state differs')
    coefficient=objective_coefficients(captured['real_targets'],captured['valid_anchors'],0.,1)
    original=MixingOnlyFutureSpanCandidate(c,span,initialize=False,fp16_body=False);original.load_state_dict(saved['model'])
    loss,_=FutureSpanObjective(original)(*values,coefficient,2,2);loss.backward()
    expected={n:p.grad.detach() for n,p in original.named_parameters()};checks=[]
    for mode in ['mix32','mixc32','full32']:
        guards(root)
        candidate=PrecisionMixingProbe(c,span,precision_mode=mode,initialize=False,fp16_body=False);candidate.load_state_dict(saved['model'])
        value,_=FutureSpanObjective(candidate)(*values,coefficient,2,2);value.backward()
        delta=gradient_comparison(expected,{n:p.grad.detach() for n,p in candidate.named_parameters()})
        if not torch.equal(value.detach(),loss.detach()) or delta['overall']['relative_l2']>1e-6:
            raise AssertionError('Precision branch changes the FP32 model function')
        if digest(dict(sorted(candidate.state_dict().items())))!=digest(dict(sorted(original.state_dict().items()))):
            raise AssertionError('Precision branch changes trained parameters')
        checks.append(dict(mode=mode,loss_exact=True,gradient=delta));del candidate,value
    if time.time()-started>180:raise TimeoutError('CPU implementation screen cap')
    if managed_running():raise RuntimeError('Failed predecessor workers must be fully closed')
    if shutil.disk_usage(root).free<50*1024**3:raise RuntimeError('Projected diagnostic storage plus protected reserve required')
    worker,host=split_snapshot_sources(sources)
    phases=[IDENTITY+'-'+mode for mode in ['mix32','full32','mixc32']]
    plan=dict(identity=IDENTITY,registered_unix=time.time(),purpose='zero_update_mixing_precision_diagnosis',
        source_sha256=sources,worker_source_sha256=worker,host_only_source_sha256=host,point=point,
        root_comm_ports=reserve_ports(root,phases),seconds_limit=2700,phase_seconds_limit=600,cpu_phase_seconds_limit=1200,
        measured_budget=dict(conservative_native_seconds_per_capture=600,independent_cpu_seconds=1200,
            predicted_seconds_with_margin=2580,cap_seconds=2700),cpu_screen=checks,cpu_preparation_seconds=time.time()-started,
        original_failure_sha256=file_sha256(failed),test_accessed=False,terminal_alpha_spent=0,
        scientific_fit_admitted=False,scope=Path(DOC).read_text())
    out.mkdir();PUBLIC.mkdir(parents=True);write_state(out/'plan.json',plan);write_state(PUBLIC/'plan.json',plan)
    write_state(PUBLIC/'status.json',dict(status='PREPARED_NO_NATIVE_YET',assigned_chips=0));return plan


class CheckedExecution(Execution):
    def check(self):
        super().check();require_inode_reserve([self.root,Path('/')])
        point=self.plan['point']
        if file_sha256(self.root/point['parent_plan'])!=point['parent_plan_sha256']:raise AssertionError('Original trained plan changed')


def run(root,resume=False):
    out=root/IDENTITY;file=out/'plan.json';plan=read(file)
    if (out/'report.json').exists():raise FileExistsError('Preserve terminal precision diagnosis')
    execution=CheckedExecution(root,file,out/'execution',resume=resume);result=dict(status='STOPPED_MIXING_PRECISION_DIAGNOSIS');captures={};comparisons={}
    def interrupted(number,frame):execution.failed.set();raise KeyboardInterrupt('Precision diagnosis interrupted')
    handlers={s:signal.signal(s,interrupted) for s in [signal.SIGINT,signal.SIGTERM]}
    def capture(mode):
        name=IDENTITY+'-'+mode;opts=['--root','/work/instance-nvme','--plan','/work/instance-nvme/'+IDENTITY+'/plan.json',
            '--plan-sha',execution.sha,'--variant',mode,'--comm-port',str(plan['root_comm_ports'][name])]
        value=execution.native(name,'scripts.capture_mixing_precision',opts,'CAPTURED_ZERO_UPDATE_FUTURE_SPAN_GRADIENT')
        if value['probe_plan_sha256']!=execution.sha or value['precision_override']!=mode:raise AssertionError('Precision capture identity differs')
        return name,value
    def compare(mode,gold=None):
        target=out/'cpu'/mode;opts=['--root',str(root),'--probe-plan',str(file),'--probe-sha',execution.sha,
            '--capture',str(root/'runs'/captures[mode][0]),'--output',str(target)]
        if gold:opts+=['--gold',str(out/'cpu'/gold),'--gold-report-sha',file_sha256(out/'cpu'/gold/'report.json')]
        return execution.cpu(mode+'-reference','scripts.compare_mixing_precision_cpu',opts,target/'report.json',
                             'COMPLETE_MIXING_PRECISION_REFERENCE_COMPARISON')
    try:
        write_state(PUBLIC/'status.json',dict(status='RUNNING_ZERO_UPDATE_PRECISION_DIAGNOSIS',maximum_chips=16))
        with cf.ThreadPoolExecutor(2) as pool:
            futures={pool.submit(capture,mode):mode for mode in ['mix32','full32']}
            for future in cf.as_completed(futures):captures[futures[future]]=future.result()
        comparisons['mix32']=compare('mix32');comparisons['full32']=compare('full32','mix32')
        if not comparisons['mix32']['passes_original_gate'] and comparisons['full32']['passes_original_gate']:
            captures['mixc32']=capture('mixc32');comparisons['mixc32']=compare('mixc32','mix32')
        execution.check();result=dict(status='COMPLETE_MIXING_PRECISION_DIAGNOSIS',comparisons=comparisons,
            optimizer_updates=0,scientific_fit_admitted=False,scope=plan['scope'])
    except BaseException as error:execution.failed.set();result.update(error=str(error),comparisons=comparisons);raise
    finally:
        execution.close(PUBLIC/'execution-costs.json')
        for number,handler in handlers.items():signal.signal(number,handler)
        result.update(plan_sha256=execution.sha,test_accessed=False,terminal_alpha_spent=0,terminal_success=False)
        write_state(out/'report.json',result);write_state(PUBLIC/'report.json',result);write_state(PUBLIC/'status.json',dict(status=result['status'],assigned_chips=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    prepare(a.root)
    if not a.prepare_only:run(a.root,a.resume)


if __name__=='__main__':main()
