"""Read every prespecified structural contrast after numerical qualification."""
import argparse,json,time
from pathlib import Path
from compute_trainium.structural_probability_statistics import structural_statistics
from compute_trainium.training import file_sha256

def main(a):
    started=time.time();p=json.loads(a.plan.read_text());folder=a.root/'runs'/p['readout_phase'];r=json.loads((folder/'report.json').read_text())
    if r['status']!='COMPLETE_STRUCTURAL_PROBABILITY_READOUT':raise AssertionError('Audited readout first')
    rows=[row for rank in range(64) for row in json.loads((folder/f'PRIVATE-rank{rank}-scores.json').read_text())]
    result=structural_statistics(rows,p['bootstrap_replicates'],p['bootstrap_seed'],p['practical_gain_nats'])
    if result['source_groups']!=p['source_groups']:raise AssertionError('Complete frozen panel required')
    result.update(status='COMPLETE_STRUCTURAL_PROBABILITY_ANALYSIS',trained_seeds=1,new_fits=0,protected_groups_scored=0,automatic_promotion=False,optimizer_updates=0,assigned_chips=0,wall_seconds=time.time()-started,score_sha256={f'PRIVATE-rank{rank}-scores.json':file_sha256(folder/f'PRIVATE-rank{rank}-scores.json') for rank in range(64)})
    a.output.mkdir(exist_ok=False);(a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'gates':result['screening_gates']}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
