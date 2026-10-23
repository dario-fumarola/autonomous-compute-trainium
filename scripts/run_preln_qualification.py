"""Bounded sequential qualification; no scientific learning claims."""
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    root=Path(os.environ['RESEARCH_ROOT'])
    limit=time.time()+900
    tiny=root/'logs/preln-tiny-ddp64-001.supervisor.json'
    while not tiny.exists():
        if time.time()>limit:raise TimeoutError('Tiny qualification did not finish')
        time.sleep(2)
    receipt=json.loads(tiny.read_text())
    if receipt['worker_exit_code'] or receipt['supervisor_failure']:raise ValueError('Tiny gate failed')
    common=['--size','small','--architecture','preln','--repetitions','2','--context','1024','--batch','2','--accumulation','2',
            '--warmup','4','--corpus','/work/natural-v2-1050m/source.json','--packed-corpus','/work/natural-v2-1050m/packed',
            '--dev-corpus','/work/natural-v2-1050m/dev.json','--data-seed','23','--save']
    for name,steps,resume in [('preln-150m-ddp64-001',104,None),
                              ('preln-150m-restore64-001',8,'/work/instance-nvme/runs/preln-150m-ddp64-001/checkpoint.pt')]:
        env=dict(os.environ,RESEARCH_TRAIN='1',RESEARCH_RUN=name,RESEARCH_DDP_RANKS='64',
                 NEURON_RT_VISIBLE_CORES=','.join(map(str,range(64))),RESEARCH_TIMEOUT_SECONDS='1500')
        args=[*common,'--steps',str(steps)]
        if resume:args += ['--resume',resume]
        subprocess.run(['python3','-m','scripts.supervise',*args],env=env,check=True)
    a=json.loads((root/'runs/preln-150m-ddp64-001/report.json').read_text())
    b=json.loads((root/'runs/preln-150m-restore64-001/report.json').read_text())
    assert a['warm_updates']>=100 and a['state_digest']==b['recovered_model_optimizer_digest']
    (root/'preln-qualification.json').write_text(json.dumps({'status':'PASS_PRELN_NATIVE_QUALIFICATION',
        'warm_real_targets_per_second':a['warm_global_real_targets_per_second'],
        'full_recovered_digest':b['recovered_model_optimizer_digest'],'ended_unix':time.time()},indent=2)+'\n')


if __name__=='__main__':main()
