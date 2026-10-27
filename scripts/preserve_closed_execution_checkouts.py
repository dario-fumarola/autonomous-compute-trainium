"""Preserve duplicate closed source checkouts exactly before releasing inodes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

from compute_trainium.execution_isolation import managed_running
from compute_trainium.training import file_sha256

NAMES=['workspace-native-fixture-worktree-v1','workspace-native-full-model-worktree-v1',
       'workspace-native-full-model-worktree-v2','workspace-native-full-model-worktree-v3',
       'workspace-native-full-update-worktree-v1','workspace-native-online-throughput-worktree-v1']


def run(root,*,scale_closed=False,recovery=False):
    if managed_running():raise RuntimeError('Closed execution checkout maintenance only')
    out=root/('workspace-scale-closed-checkout-preservation-v2' if recovery else 'workspace-scale-closed-checkout-preservation-v1' if scale_closed else 'workspace-closed-checkout-preservation-v1');out.mkdir(mode=0o700)
    public=Path('results/workspace_scale_closed_checkouts_20261027/recovery-v2' if recovery else 'results/workspace_scale_closed_checkouts_20261027' if scale_closed else 'results/workspace_closed_checkouts_20261027');public.mkdir()
    started=time.time();before=os.statvfs(root).f_favail;rows=[]
    for name in NAMES:
        tree=root/name
        if not tree.is_dir():raise FileNotFoundError('No unregistered checkout substitution')
        for process in Path('/proc').iterdir():
            if not process.name.isdigit():continue
            try:
                if Path(os.readlink(process/'cwd'))==tree:raise RuntimeError('Checkout still owns a process')
            except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        commit=subprocess.check_output(['git','-C',str(tree),'rev-parse','HEAD'],text=True).strip()
        subprocess.run(['git','-C',str(tree),'diff','--exit-code'],check=True,capture_output=True)
        files=[p for p in tree.rglob('*') if p.is_file() and not p.is_symlink()]
        if any(p.suffix in ['.pt','.npy','.safetensors','.bin'] for p in files):raise RuntimeError('Private training artifacts must not be treated as duplicate source')
        expected={str(p.relative_to(root)):file_sha256(p) for p in files}
        archive=out/(name+'.tar.gz')
        with tarfile.open(archive,'x:gz',compresslevel=1) as target:target.add(tree,arcname=name,recursive=True)
        found={}
        with tarfile.open(archive,'r:gz') as saved:
            for member in saved:
                if member.isfile():found[member.name]=hashlib.sha256(saved.extractfile(member).read()).hexdigest()
        if found!=expected:raise AssertionError('Complete source/receipt byte coverage differs')
        row=dict(name=name,commit=commit,files=len(files),archive_sha256=file_sha256(archive),archive_bytes=archive.stat().st_size,exact_source_and_receipt_coverage=True)
        (out/(name+'-verified.json')).write_text(json.dumps(row,indent=2)+'\n')
        subprocess.run(['git','worktree','remove','--force',str(tree)],check=True,capture_output=True)
        rows.append(row)
    result=dict(status='COMPLETE_EXACT_CLOSED_CHECKOUT_PRESERVATION',checkouts=rows,free_inodes_before=before,
        free_inodes_after=os.statvfs(root).f_favail,wall_seconds=time.time()-started,
        scope='Only listed duplicated closed execution source checkouts.No source snapshot,run receipt,checkpoint,optimizer,gradient,corpus or current/future execution checkout removed.Full per-file SHA coverage verified in private tar archives before git worktree removal.No Native worker or optimizer ran during this maintenance.')
    for d in [out,public]:(d/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['free_inodes_before'],result['free_inodes_after'])


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--scale-closed',action='store_true');p.add_argument('--recovery',action='store_true');a=p.parse_args()
    if a.scale_closed:
        NAMES=['scale-feasibility-worktree-v1','scale-model-worktree-v1','scale-runtime-worktree-v1',
            'scale-audit-worktree-v1','scale-body-main-handoff-worktree-v1','scale-final-lower-rate-recovery-worktree-v1']
    if a.recovery:
        NAMES=['scale-audit-worktree-v1','scale-body-main-handoff-worktree-v1','scale-final-lower-rate-recovery-worktree-v1',
            'workspace-stable-loss-fixture-worktree-v1','workspace-stable-model-worktree-v1']
    run(a.root,scale_closed=a.scale_closed,recovery=a.recovery)
