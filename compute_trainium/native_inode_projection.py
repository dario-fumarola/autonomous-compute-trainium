"""Conservative source-copy and artifact inode admission before native phases."""
from pathlib import Path


def project_snapshot_inodes(repo,phases,*,reserve=10000,artifacts_per_phase=512,compiler_allowance=5000):
    if type(phases) is not int or not 1<=phases<=16 or reserve<10000:raise ValueError('Bounded phases and protected inode reserve required')
    files=directories=python_files=0
    for name in ['compute_trainium','scripts','configs','third_party']:
        folder=Path(repo)/name
        if not folder.exists():continue
        directories+=1
        for path in folder.rglob('*'):
            if path.is_file():files+=1;python_files+=path.suffix=='.py'
            elif path.is_dir():directories+=1
    # Keep existing bytecode copied with the tree,and allow one new native
    # bytecode file plus a directory per Python source,even if not imported.
    per_phase=files+directories+2*python_files+artifacts_per_phase
    projected=phases*per_phase+compiler_allowance
    return dict(phases=phases,source_files=files,source_directories=directories,python_sources=python_files,
        artifacts_per_phase=artifacts_per_phase,compiler_allowance=compiler_allowance,
        projected_new_inodes=projected,protected_reserve=reserve,required_free_inodes=reserve+projected)
