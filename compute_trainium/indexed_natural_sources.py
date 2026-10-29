"""Virtual packed natural tokens over immutable original source arrays.

The index records document ranges; it never contains generated source tokens.
Virtual BOS/EOS positions receive no target weight. Adjacent windows share one
input position, while each real next-token label occurs exactly once.
"""
import json
from pathlib import Path
import numpy as np
import torch

RECORD_DTYPE=np.dtype([('start','<u8'),('offset','<u8'),('length','<u4'),('domain','u1')])
DOMAINS=('web','math','code')

class IndexedNaturalSources:
    def __init__(self,root):
        self.root=Path(root);self.report=json.loads((self.root/'report.json').read_text())
        if self.report['status']!='PASS_INDEXED_NATURAL_SOURCES':raise ValueError('Accepted indexed source receipt required')
        self.context=self.report['context'];self.windows=self.report['windows']
        self.records=np.load(self.root/'PRIVATE-documents.npy',mmap_mode='r')
        if self.records.dtype!=RECORD_DTYPE:raise ValueError('Exact source-index schema required')
        if len(self.records)==0 or self.records['start'][0]!=0:raise ValueError('Nonempty contiguous source index required')
        if not np.all(self.records['start'][1:]==self.records['start'][:-1]+self.records['length'][:-1]+2):raise ValueError('Packed document gaps or overlaps')
        if not np.all((self.records['domain']>=1)&(self.records['domain']<=3)):raise ValueError('Unknown source domain')
        source=self.root.parent/self.report['source_relative']
        self.arrays={i+1:np.memmap(source/(d+'.u16'),dtype='<u2',mode='r') for i,d in enumerate(DOMAINS)}
        # Searchsorted on a strided structured field copies the entire index
        # on every call. Materialize this small scalar column once per worker.
        self.starts=np.ascontiguousarray(self.records['start'])
        self.cumulative=np.concatenate((np.zeros(1,dtype=np.uint64),np.cumsum(self.records['length'],dtype=np.uint64)))
        self.valid_positions=int(self.records['start'][-1])+int(self.records['length'][-1])+2
        if self.valid_positions!=self.report['valid_positions']:raise ValueError('Source index receipt mismatch')

    def prefix_targets(self,position):
        """Real targets in virtual positions [0, position)."""
        if position<0:raise ValueError('Negative position')
        if position>=self.valid_positions:return int(self.cumulative[-1])
        j=int(np.searchsorted(self.starts,np.uint64(position),side='right'))-1
        if j<0:return 0
        row=self.records[j];within=max(0,min(int(row['length']),position-int(row['start'])-1))
        return int(self.cumulative[j])+within

    def target_count(self,start,count):
        if min(start,count)<0 or start+count>self.windows:raise ValueError('Window range exceeds admitted source')
        a=start*self.context+1;b=(start+count)*self.context+1
        return self.prefix_targets(b)-self.prefix_targets(a)

    def window(self,index):
        if not 0<=index<self.windows:raise IndexError('Window exceeds admitted source')
        begin=index*self.context;end=begin+self.context+1
        tokens=np.full(self.context+1,2,dtype=np.int64);domains=np.zeros(self.context+1,dtype=np.uint8)
        j=max(0,int(np.searchsorted(self.starts,np.uint64(begin),side='right'))-1)
        while j<len(self.records) and int(self.records[j]['start'])<end:
            r=self.records[j];start=int(r['start']);length=int(r['length']);domain=int(r['domain'])
            if begin<=start<end:tokens[start-begin]=1
            lo=max(begin,start+1);hi=min(end,start+1+length)
            if hi>lo:
                offset=int(r['offset'])+lo-start-1
                values=self.arrays[domain][offset:offset+hi-lo]
                if len(values)!=hi-lo:raise ValueError('Original source offset exceeds array')
                tokens[lo-begin:hi-begin]=values;domains[lo-begin:hi-begin]=domain
            j+=1
        return tokens,domains

    def batch(self,indices):
        if not indices:raise ValueError('Nonempty batch required')
        views=[self.window(int(i)) for i in indices]
        tokens=np.stack([x for x,_ in views]);domains=np.stack([d for _,d in views])
        return (torch.from_numpy(tokens[:,:-1].copy()),torch.from_numpy(tokens[:,1:].copy()),
                torch.from_numpy((domains[:,1:]!=0).copy()))
