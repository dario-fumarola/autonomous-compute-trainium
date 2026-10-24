"""Reuse the full qualification objective with changing runtime scalars."""
import json
import os
import sys
from pathlib import Path

from compute_trainium.parcae_runtime_scalars import ParcaeRuntimeScalarOptimizer,changing_qualification_schedule
from scripts import train_parcae_precise_qualification as qualified


def main():
    if '--comm-port' not in sys.argv:raise ValueError('Explicit managed root port required')
    index=sys.argv.index('--comm-port');port=int(sys.argv[index+1]);del sys.argv[index:index+2]
    if not 16000<=port<30000:raise ValueError('Registered non-ephemeral root port required')
    os.environ['NEURON_RT_ROOT_COMM_ID']='127.0.0.1:'+str(port)
    qualified.ParcaeOptimizerCandidate=ParcaeRuntimeScalarOptimizer
    qualified.configure_optimizer=changing_qualification_schedule
    qualified.main()
    if int(os.environ.get('RANK','0'))==0:
        path=Path(sys.argv[sys.argv.index('--output')+1])/'report.json'
        row=json.loads(path.read_text());row.update(scalar_input_policy='runtime_host_f32_tensor_copy',
            warm_schedule='100 distinct LR and cautious-decay values; same first20 conservative updates')
        path.write_text(json.dumps(row,indent=2)+'\n')


if __name__=='__main__':main()
