"""Independent hard-deadline guard for this dedicated research container."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path


def main():
 p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);a=p.parse_args()
 plan=json.loads((a.campaign/'plan.json').read_text());receipt=a.campaign/'deadline-guard.json'
 while time.time()<plan['deadline_unix']:
  if (a.campaign/'queue/report.json').exists():
   receipt.write_text(json.dumps({'status':'QUEUE_FINISHED_BEFORE_DEADLINE','unix':time.time()})+'\n');return
  time.sleep(min(30,max(0,plan['deadline_unix']-time.time())))
 subprocess.run(['docker','stop','--time','10','autonomous-compute-neuron233'],capture_output=True,timeout=30)
 state=json.loads((a.campaign/'queue/state.json').read_text())
 for stage in state['stages'].values():
  if stage.get('status')=='RUNNING' and stage.get('pid'):
   try:os.killpg(stage['pid'],signal.SIGTERM)
   except ProcessLookupError:pass
 receipt.write_text(json.dumps({'status':'STOPPED_AT_DECLARED_CAMPAIGN_DEADLINE','unix':time.time()})+'\n')


if __name__=='__main__':main()
