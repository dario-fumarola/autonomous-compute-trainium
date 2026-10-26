"""Recover zero-filled blocks of closed checkpoints without changing bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from compute_trainium.inode_reserve import require_inode_reserve
from compute_trainium.queue import write_state
from compute_trainium.training import file_sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--apply',action='store_true')
    parser.add_argument('--max-gib',type=int,default=256);parser.add_argument('--target-free-gib',type=int,default=20)
    parser.add_argument('--seconds-limit',type=int,default=900);a=parser.parse_args();started=time.time()
    if not 0<a.max_gib<=256 or not 0<a.target_free_gib<=20 or not 0<a.seconds_limit<=900:
        raise ValueError('Bounded storage maintenance only')
    proof=json.loads((a.root/'zero-storage-proof-v1/report.json').read_text())
    if (proof['status']!='PASS_BYTE_EXACT_ZERO_STORAGE_PROOF' or not proof['byte_sha256_unchanged']
            or not proof['permissions_and_owners_unchanged'] or proof['reclaimed_bytes']<=0):
        raise ValueError('Actual same-filesystem byte-preservation proof required')
    a.output.mkdir(parents=True,exist_ok=False);candidates=[]
    for folder in sorted((a.root/'runs').iterdir()):
        # The entire live campaign stays outside this maintenance operation,
        # including a stable prefix that happens to finish during the scan.
        if folder.name.startswith('fixed-control-replication-main-v1'):continue
        receipt=folder/'report.json';closure=a.root/'snapshots'/folder.name/'closure.json'
        if not receipt.is_file() or not closure.is_file():continue
        report=json.loads(receipt.read_text())
        if (report.get('status')!='COMPLETE_REGISTERED_FUTURE_SPAN_SCIENTIFIC_SEGMENT'
                or report.get('objective_weight')!=0. or json.loads(closure.read_text())['exit_code']!=0):continue
        for row in report['checkpoint_records']:
            if type(row['update']) is not int or row['update']<1:raise ValueError('Invalid checkpoint identifier')
            file=folder/f'checkpoint-u{row["update"]}.pt'
            if file.is_symlink() or not file.is_file():continue
            stat=file.stat()
            if stat.st_size<500*1024**2 or stat.st_blocks*512<.999*stat.st_size:continue
            candidates.append(dict(file=str(file.relative_to(a.root)),sha256=row['checkpoint_sha256'],
                                   size=stat.st_size,receipt_sha256=file_sha256(receipt),closure_sha256=file_sha256(closure)))
    write_state(a.output/'manifest.json',dict(candidates=candidates,source_sha256=file_sha256(__file__),
        max_bytes=a.max_gib*1024**3,target_reclaimed_bytes=a.target_free_gib*1024**3,
        seconds_limit=a.seconds_limit,apply=a.apply,proof_sha256=file_sha256(a.root/'zero-storage-proof-v1/report.json')))
    records=[];scanned=reclaimed=0;result=dict(status='LISTED_COMPLETED_CHECKPOINT_ZERO_STORAGE')
    try:
        if a.apply:
            for candidate in candidates:
                if reclaimed>=a.target_free_gib*1024**3 or scanned+candidate['size']>a.max_gib*1024**3:break
                if time.time()-started>=a.seconds_limit:break
                require_inode_reserve([a.root,Path('/')])
                available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
                if available<512*1024**3 or min(shutil.disk_usage(a.root).free,shutil.disk_usage('/').free)<35*1024**3:
                    raise RuntimeError('Protected shared reserves failed')
                file=a.root/candidate['file'];before=file.stat()
                if file_sha256(file)!=candidate['sha256']:raise AssertionError('Completed checkpoint checksum changed')
                command=['sudo','-n','ionice','-c','3','fallocate','--dig-holes','--keep-size','--',str(file)]
                mark=time.time();error=None
                try:subprocess.run(command,check=True,timeout=min(180,max(1,a.seconds_limit-(time.time()-started))))
                except BaseException as exception:error=exception
                # Even interrupted filesystem work must prove byte preservation.
                after=file.stat();observed=file_sha256(file)
                if (observed!=candidate['sha256'] or any(getattr(before,k)!=getattr(after,k)
                        for k in ['st_size','st_ino','st_mode','st_uid','st_gid'])
                        or after.st_blocks>before.st_blocks):
                    raise AssertionError('Byte/identity/permission preservation failed')
                saved=(before.st_blocks-after.st_blocks)*512;scanned+=before.st_size;reclaimed+=saved
                records.append(dict(file=candidate['file'],sha256=observed,size_bytes=before.st_size,
                    before_allocated_bytes=before.st_blocks*512,after_allocated_bytes=after.st_blocks*512,
                    reclaimed_bytes=saved,byte_identity_preserved=True,wall_seconds=time.time()-mark,
                    error=str(error) if error else None))
                write_state(a.output/'progress.json',dict(files=len(records),scanned_bytes=scanned,reclaimed_bytes=reclaimed,records=records))
                print(json.dumps(dict(files=len(records),scanned_gib=scanned/2**30,reclaimed_gib=reclaimed/2**30)),flush=True)
                if error:raise error
            result['status']='COMPLETE_BYTE_EXACT_CHECKPOINT_ZERO_STORAGE'
    except BaseException as error:result.update(status='STOPPED_CHECKPOINT_ZERO_STORAGE',error=str(error));raise
    finally:
        result.update(records=records,scanned_bytes=scanned,reclaimed_bytes=reclaimed,
            source_sha256=file_sha256(__file__),wall_seconds=time.time()-started,assigned_chips=0,
            model_forwards=0,optimizer_updates=0,test_accessed=False,
            scope='Zero-filled blocks only;all bytes,logical sizes,paths,inodes,owners,permissions and recorded scientific hashes retained. Only successfully closed zero-auxiliary-weight scientific checkpoints;whole active campaign excluded. CPU maintenance overlaps native training and its elapsed time is not added to native reservation costs.')
        write_state(a.output/'report.json',result)


if __name__=='__main__':main()
