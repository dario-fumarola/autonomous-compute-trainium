"""Original random C0 for a declared development phase, zero model forwards."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['tune','main'],required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();started=time.time()
    import torch
    from compute_trainium.scale_pilot_size import profiled_inventory
    from compute_trainium.parcae_candidate import ParcaeCandidate,ParcaeCandidateConfig
    from compute_trainium.training import atomic_checkpoint,file_sha256
    if (a.phase,a.seed) not in [('tune',38213),('main',49177)]:raise ValueError('Prespecified development seeds/phases required')
    if a.output.exists():raise FileExistsError('Preserve all original initializer receipts')
    inventories=profiled_inventory('medium');bytes_needed=sum(r['parameters']*4 for r in inventories)+512*1024**2
    if shutil.disk_usage(a.output.parent).free<35*1024**3+bytes_needed:raise RuntimeError('Complete C0 inventory plus reserve required')
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if available<544*1024**3:raise RuntimeError('CPU construction headroom required')
    torch.set_num_threads(24);a.output.mkdir();records=[]
    for row in inventories:
        torch.manual_seed(a.seed);config=ParcaeCandidateConfig(**row['config']);recurrent=row['role']=='fixed_two'
        model=ParcaeCandidate(config,recurrent=recurrent,initialize=True)
        if sum(p.numel() for p in model.parameters())!=row['parameters']:raise AssertionError('Initializer inventory changed')
        file=a.output/(row['role']+'.pt')
        atomic_checkpoint(file,dict(purpose='scale_registered_unupdated_initializer',phase=a.phase,seed=a.seed,
            eligible_for_science=True,qualification_only=False,model_forwards=0,optimizer_updates=0,
            config=vars(config),recurrent=recurrent,model=model.state_dict()))
        records.append(dict(role=row['role'],parameters=row['parameters'],file=file.name,bytes=file.stat().st_size,sha256=file_sha256(file)))
        del model;gc.collect()
    report=dict(status='PREPARED_ORIGINAL_RANDOM_SCALE_TRAINING_INITIALIZERS',phase=a.phase,seed=a.seed,records=records,
        optimizer_updates=0,model_forwards=0,assigned_chips=0,wall_seconds=time.time()-started)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
