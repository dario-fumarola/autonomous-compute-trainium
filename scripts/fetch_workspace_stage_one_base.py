"""Acquire a pinned discovery base directly,without a second weight cache."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import requests

REPOSITORY='allenai/OLMo-2-0425-1B'
REVISION='9d3e43659f00c17e6da23cf32333afd1fc39fa1a'
BRANCH='stage1-step1907359-tokens4001B'
WEIGHTS={'model-00001-of-00002.safetensors':'7e4abea26202188a7b62c69f5274536a8a8e1e5a0f2d72fa53c68b7c8fc2f0e5',
         'model-00002-of-00002.safetensors':'e380e1d8cc9ef28c75f44b954fe59566bc144d924f3a235a8c6d7bcb040696fd'}
SMALL={'config.json','generation_config.json','model.safetensors.index.json',
       'special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt'}
PUBLIC=Path('results/workspace_base_preparation_20261027')


def fetch(root):
    out=root/'workspace-olmo2-stage1-base-v1'
    if out.exists() or PUBLIC.exists():raise FileExistsError('Preserve every base acquisition attempt')
    started=time.time();session=requests.Session()
    response=session.get('https://huggingface.co/api/models/'+REPOSITORY+'/revision/'+REVISION,
                         params={'blobs':'true'},timeout=30)
    if response.status_code!=200:raise RuntimeError('Pinned metadata HTTP '+str(response.status_code))
    metadata=response.json()
    if metadata['sha']!=REVISION:raise AssertionError('Immutable base revision changed')
    rows=[r for r in metadata['siblings'] if r['rfilename'] in SMALL or r['rfilename'] in WEIGHTS]
    if set(r['rfilename'] for r in rows)!=SMALL|WEIGHTS.keys():raise ValueError('Complete exact base inventory required')
    total=sum(r['size'] for r in rows)
    if total>7*1024**3 or shutil.disk_usage(root).free<36*1024**3+total:
        raise RuntimeError('Complete bounded download plus filesystem reserve required')
    out.mkdir(mode=0o700);PUBLIC.mkdir()
    plan=dict(repository=REPOSITORY,revision=REVISION,upstream_branch=BRANCH,expected_bytes=total,
        registered_unix=started,maximum_seconds=1800,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        pretrained_discovery_only=True,training_updates=0,assigned_chips=0,
        provenance='Stage-one checkpoint; excludes later stage-two mixture. Upstream provenance is documented,not a complete independent corpus audit.')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');(PUBLIC/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    records=[];result=dict(status='STOPPED_WORKSPACE_BASE_ACQUISITION',optimizer_updates=0,assigned_chips=0)
    try:
        for row in rows:
            name=row['rfilename'];path=out/(name+'.incomplete');digest=hashlib.sha256();blob=hashlib.sha1()
            blob.update(('blob '+str(row['size'])+'\0').encode());received=0
            if time.time()-started>1800:raise TimeoutError('Bounded base acquisition ended')
            if shutil.disk_usage(root).free<35*1024**3+row['size']:raise RuntimeError('Protected download reserve')
            with session.get('https://huggingface.co/'+REPOSITORY+'/resolve/'+REVISION+'/'+name,stream=True,timeout=(30,30)) as response:
                if response.status_code!=200:raise RuntimeError('Asset HTTP '+str(response.status_code)+' for '+name)
                with path.open('xb') as output:
                    for chunk in response.iter_content(4*1024**2):
                        if time.time()-started>1800:raise TimeoutError('Bounded download watchdog')
                        received+=len(chunk)
                        if received>row['size']:raise ValueError('Asset exceeds registered length')
                        output.write(chunk);digest.update(chunk);blob.update(chunk)
                        if shutil.disk_usage(root).free<35*1024**3:raise RuntimeError('Protected output reserve')
                if received!=row['size']:raise AssertionError('Incomplete asset length')
            if name in WEIGHTS:
                if digest.hexdigest()!=WEIGHTS[name] or row['lfs']['sha256']!=WEIGHTS[name]:
                    raise AssertionError('Pinned original weight SHA256 differs')
            elif blob.hexdigest()!=row['blobId']:raise AssertionError('Pinned Git blob differs')
            path.rename(out/name);records.append(dict(file=name,bytes=received,sha256=digest.hexdigest()))
            (out/'progress.json').write_text(json.dumps(dict(files=len(records),total_files=len(rows),downloaded_bytes=sum(r['bytes'] for r in records))))
        result['status']='COMPLETE_PINNED_STAGE_ONE_BASE_ACQUISITION'
    except BaseException as error:
        result.update(error_type=type(error).__name__);raise
    finally:
        result.update(records=records,wall_seconds=time.time()-started,revision=REVISION)
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');(PUBLIC/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();fetch(a.root)
