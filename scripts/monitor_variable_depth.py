"""Independent liveness/deadline monitor; no model choice or automatic retuning."""
import argparse
import datetime as dt
import json
import subprocess
import time
from pathlib import Path
from compute_trainium.queue import write_state


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--pid',type=int,required=True)
    a=p.parse_args();deadline=dt.datetime(2026,10,24,4,15,tzinfo=dt.timezone.utc).timestamp()
    while not (a.campaign/'report.json').exists():
        if not Path('/proc/'+str(a.pid)).exists():
            write_state(a.campaign/'liveness.json',{'status':'CONTROLLER_EXITED_WITHOUT_RESULT','unix':time.time()});return
        if time.time()>deadline:
            subprocess.run(['docker','stop','--time','5','autonomous-compute-neuron233'],capture_output=True,timeout=25)
            write_state(a.campaign/'liveness.json',{'status':'DEDICATED_WORKER_STOPPED_AT_HARD_DEADLINE','unix':time.time()});return
        workers=0
        for p in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                parts=p.read_bytes().split(b'\0')
                workers+=b'scripts.train_variable_depth' in parts or b'scripts.evaluate_depth_curve' in parts
            except (FileNotFoundError,PermissionError):pass
        write_state(a.campaign/'liveness.json',{'status':'CONTROLLER_ALIVE','unix':time.time(),'native_workers':workers})
        time.sleep(20)
    write_state(a.campaign/'liveness.json',{'status':'CAMPAIGN_CLOSED','unix':time.time()})


if __name__=='__main__':main()
