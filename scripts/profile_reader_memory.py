"""Bounded read-only resource profile; no model scoring or source selection."""
import argparse
import gc
import itertools
import json
import os
import time
from pathlib import Path

import pyarrow as pa

from scripts.prepare_natural import rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-map',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--strategy',choices=['interleaved','sequential'],required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    paths=[Path(d['path']) for d in json.loads(a.source_map.read_text()) if d['domain']=='web'][:16]
    def rss():return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    baseline=rss();started=time.time();opened=[];samples=[]
    try:
        for i,path in enumerate(paths):
            reader=iter(rows(path));opened.append(reader)
            values=list(itertools.islice(reader,256));del values
            if a.strategy=='sequential':
                reader.close();opened.pop();gc.collect();pa.default_memory_pool().release_unused()
            samples.append({'files_read':i+1,'open_readers':len(opened),
                            'rss_bytes':rss(),'active_arrow_bytes':pa.total_allocated_bytes()})
            if rss()-baseline>2*1024**3 or time.time()-started>90:break
    finally:
        for reader in opened:reader.close()
    a.output.write_text(json.dumps({'status':'COMPLETE_BOUNDED_READER_RESOURCE_PROFILE',
        'strategy':a.strategy,'baseline_rss_bytes':baseline,'samples':samples,
        'wall_seconds':time.time()-started,'scope':'First256 rows of at most16 previously pinned web files. Stop after2GiB RSS growth or90s. No model, optimizer, dataset admission or TEST.'},indent=2)+'\n')


if __name__=='__main__':main()
