"""Atomic receipts for a finite, dependency-ordered research queue."""
import json
from pathlib import Path


def write_state(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def prerequisites(stage,states):
    deps=stage.get('dependencies',[])
    blocked=[d for d in deps if states.get(d,{}).get('status') in ['FAILED','BLOCKED','DEADLINE']]
    if blocked:return 'BLOCKED',blocked
    if all(states.get(d,{}).get('status')=='COMPLETE' for d in deps):return 'READY',[]
    return 'WAITING',[]
