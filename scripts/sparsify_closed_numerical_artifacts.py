"""Preserve closed tensor artifacts byte-for-byte while reclaiming zero blocks."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

from compute_trainium.execution_isolation import managed_running
from compute_trainium.inode_reserve import require_inode_reserve
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256

ROLES={'gradient.pt':'gradient_sha256','initial-global-native.pt':'initial_global_gradient_sha256',
       'updated-global-native.pt':'updated_global_gradient_sha256','after-1-model-optimizer.pt':'first_checkpoint_sha256',
       'after-121-model-optimizer.pt':'final_checkpoint_sha256','after-120-full-state.pt':'checkpoint_sha256',
       'after-first-update.pt':'first_update_sha256'}


def recorded_roles(report):
    return {name:report[key] for name,key in ROLES.items() if isinstance(report.get(key),str) and len(report[key])==64}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--apply',action='store_true');a=p.parse_args();started=time.time()
    if a.output.exists():raise FileExistsError('Preserve storage maintenance attempt')
    if managed_running():raise RuntimeError('No active native owners during closed-artifact maintenance')
    proof_file=a.root/'zero-storage-proof-v1/report.json';proof=json.loads(proof_file.read_text())
    if proof['status']!='PASS_BYTE_EXACT_ZERO_STORAGE_PROOF' or not proof['byte_sha256_unchanged']:raise ValueError('Existing same-filesystem proof required')
    candidates=[]
    for folder in sorted((a.root/'runs').iterdir()):
        if folder.name.startswith(('baseline-','one-control-','fixed-control-')):continue
        receipt=folder/'report.json';closure=a.root/'snapshots'/folder.name/'closure.json'
        if not receipt.is_file() or not closure.is_file() or json.loads(closure.read_text())['exit_code']!=0:continue
        for name,sha in recorded_roles(json.loads(receipt.read_text())).items():
            file=folder/name
            if not file.is_file() or file.is_symlink():continue
            stat=file.stat()
            if not 200*1024**2<=stat.st_size<=4*1024**3 or stat.st_blocks*512<.9*stat.st_size:continue
            candidates.append(dict(file=str(file.relative_to(a.root)),sha256=sha,size=stat.st_size,
                receipt_sha256=file_sha256(receipt),closure_sha256=file_sha256(closure)))
    a.output.mkdir(parents=True);write_state(a.output/'manifest.json',dict(candidates=candidates,source_sha256=file_sha256(__file__),
        max_scanned_bytes=128*1024**3,target_reclaimed_bytes=15*1024**3,seconds_limit=600,apply=a.apply,
        proof_sha256=file_sha256(proof_file)))
    records=[];scanned=reclaimed=0;result=dict(status='LISTED_CLOSED_NUMERICAL_ZERO_STORAGE')
    try:
        if a.apply:
            for item in candidates:
                if reclaimed>=15*1024**3 or scanned+item['size']>128*1024**3 or time.time()-started>=600:break
                require_inode_reserve([a.root,Path('/')])
                if min(shutil.disk_usage(a.root).free,shutil.disk_usage('/').free)<35*1024**3:raise RuntimeError('Protected byte-space reserve failed')
                file=a.root/item['file'];before=file.stat()
                if file_sha256(file)!=item['sha256']:raise AssertionError('Recorded closed artifact bytes differ')
                error=None;mark=time.time()
                try:subprocess.run(['sudo','-n','ionice','-c','3','fallocate','--dig-holes','--keep-size','--',str(file)],check=True,
                                   timeout=min(120,max(1,600-(time.time()-started))))
                except BaseException as exception:error=exception
                after=file.stat();observed=file_sha256(file)
                if observed!=item['sha256'] or any(getattr(before,k)!=getattr(after,k) for k in ['st_size','st_ino','st_mode','st_uid','st_gid']):
                    raise AssertionError('Closed tensor byte/identity preservation failed')
                if after.st_blocks>before.st_blocks:raise AssertionError('Sparse operation expanded allocation')
                saved=(before.st_blocks-after.st_blocks)*512;scanned+=before.st_size;reclaimed+=saved
                records.append(dict(file=item['file'],sha256=observed,size_bytes=before.st_size,reclaimed_bytes=saved,
                    byte_identity_preserved=True,wall_seconds=time.time()-mark,error=str(error) if error else None))
                write_state(a.output/'progress.json',dict(files=len(records),scanned_bytes=scanned,reclaimed_bytes=reclaimed,records=records))
                if error:raise error
            result['status']='COMPLETE_BYTE_EXACT_CLOSED_NUMERICAL_ZERO_STORAGE'
    except BaseException as error:result.update(status='STOPPED_CLOSED_NUMERICAL_ZERO_STORAGE',error=str(error));raise
    finally:
        result.update(records=records,scanned_bytes=scanned,reclaimed_bytes=reclaimed,wall_seconds=time.time()-started,
            source_sha256=file_sha256(__file__),assigned_chips=0,model_forwards=0,optimizer_updates=0,test_accessed=False,
            scope='Only recorded artifacts of successful closed native phases;current baseline/control histories excluded. Zero blocks only:all paths,bytes,logical sizes,inodes,owners,permissions and recorded hashes retained. No model/state/source deletion. CPU maintenance cost is separate from assigned accelerator work.')
        write_state(a.output/'report.json',result);print(json.dumps({k:result[k] for k in ['status','scanned_bytes','reclaimed_bytes','wall_seconds']}))


if __name__=='__main__':main()
