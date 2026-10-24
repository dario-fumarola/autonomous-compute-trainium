"""Independent liveness/deadline guard, with no model or success selection."""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from compute_trainium.queue import write_state


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--pid',type=int,required=True);a=p.parse_args()
    plan=json.loads((a.campaign/'plan.json').read_text())
    while not (a.campaign/'report.json').exists():
        if not Path('/proc/'+str(a.pid)).exists():
            subprocess.run(['docker','stop','--time','5','autonomous-compute-neuron233'],capture_output=True,timeout=25)
            child=a.campaign/'independent-cpu-process.json'
            if child.exists():
                pid=json.loads(child.read_text())['pid'];cmd=Path('/proc/'+str(pid)+'/cmdline')
                if cmd.exists() and str(a.campaign/'independent-cpu').encode() in cmd.read_bytes().split(b'\0'):
                    try:os.killpg(pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            report={'status':'STOPPED_CONTROLLER_EXITED_WITHOUT_RESULT','unix':time.time()}
            write_state(a.campaign/'liveness.json',report);write_state(a.campaign/'report.json',report);return
        if time.time()>plan['deadline_unix']:
            subprocess.run(['docker','stop','--time','5','autonomous-compute-neuron233'],capture_output=True,timeout=25)
            write_state(a.campaign/'liveness.json',{'status':'STOPPED_AT_HARD_DEADLINE','unix':time.time()});return
        workers=0
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            try:args=path.read_bytes().split(b'\0');workers+=b'scripts.train_variable_depth' in args or b'scripts.evaluate_depth_curve' in args
            except (FileNotFoundError,PermissionError):pass
        write_state(a.campaign/'liveness.json',{'status':'CONTROLLER_ALIVE','unix':time.time(),'native_workers':workers})
        time.sleep(20)
    write_state(a.campaign/'liveness.json',{'status':'CAMPAIGN_CLOSED','unix':time.time()})


if __name__=='__main__':main()
