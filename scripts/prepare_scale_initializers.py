"""Random initialization only: no model forwards, examples or learning updates."""
import argparse
import json
from pathlib import Path
import resource
import shutil
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=1009);args=parser.parse_args();started=time.time()
    import torch
    from compute_trainium.parcae_candidate import ParcaeCandidateConfig,ParcaeCandidate
    from compute_trainium.scale_sharded_candidate import scale_inventory
    from compute_trainium.training import atomic_checkpoint,file_sha256
    torch.set_num_threads(24)
    if args.output.exists():raise FileExistsError('Preserve every initialization attempt')
    if shutil.disk_usage(args.output.parent).free<115*1024**3:
        raise RuntimeError('Complete projected artifact allowance plus protected reserve required')
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if available<544*1024**3:raise RuntimeError('Protected host reserve plus CPU construction headroom required')
    args.output.mkdir();records=[]
    for inventory in scale_inventory():
        torch.manual_seed(args.seed)
        config=ParcaeCandidateConfig(**inventory['config']);recurrent=inventory['role']=='fixed_two'
        model=ParcaeCandidate(config,recurrent=recurrent,initialize=True)
        if sum(p.numel() for p in model.parameters())!=inventory['parameters']:
            raise AssertionError('Exact CPU inventory changed')
        file=args.output/(inventory['role']+'.pt')
        atomic_checkpoint(file,dict(purpose='scale_initializer_qualification_only',qualification_only=True,
            eligible_for_science=False,seed=args.seed,config=vars(config),recurrent=recurrent,
            parameters=inventory['parameters'],model=model.state_dict()))
        records.append(dict(role=inventory['role'],parameters=inventory['parameters'],file=file.name,
            sha256=file_sha256(file),bytes=file.stat().st_size,elapsed_seconds=time.time()-started))
        del model
        import gc,ctypes
        gc.collect();ctypes.CDLL(None).malloc_trim(0)
        (args.output/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
    report=dict(status='PREPARED_SCALE_QUALIFICATION_RANDOM_INITIALIZERS',seed=args.seed,records=records,
        model_forwards=0,optimizer_updates=0,research_examples=0,assigned_chips=0,
        wall_seconds=time.time()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scientific_fit_admitted=False)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
