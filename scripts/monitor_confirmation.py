"""Independent liveness record; never mutates the scientific queue or workers."""
import argparse
import json
import time
from pathlib import Path
from compute_trainium.queue import write_state


def inspect(controller_pid):
    native={name:0 for name in ['scripts.train_distributed','scripts.evaluate_trained_path','scripts.train_fast']}
    controller=False
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:argv=[x.decode(errors='replace') for x in p.read_bytes().split(b'\0') if x]
        except (FileNotFoundError,PermissionError):continue
        if '-m' not in argv:continue
        i=argv.index('-m')+1
        if i>=len(argv):continue
        module=argv[i]
        if p.parent.name==str(controller_pid) and module=='scripts.run_confirmation_campaign':controller=True
        if module in native:native[module]+=1
    return {'controller_alive':controller,'native_worker_counts':native}


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--controller-pid',type=int,required=True);p.add_argument('--deadline-unix',type=float,required=True);a=p.parse_args()
    while True:
        record={'recorded_unix':time.time(),**inspect(a.controller_pid)}
        done=a.campaign/'queue/report.json'
        if done.exists():record.update(status='CONTROLLER_FINISHED',result=json.loads(done.read_text())['status'])
        elif not record['controller_alive']:record['status']='CONTROLLER_MISSING'
        elif time.time()>a.deadline_unix:record['status']='DECLARED_DEADLINE_REACHED'
        else:record['status']='CONTROLLER_ALIVE'
        write_state(a.campaign/'queue/liveness.json',record)
        if record['status']!='CONTROLLER_ALIVE':return
        time.sleep(5)


if __name__=='__main__':main()
