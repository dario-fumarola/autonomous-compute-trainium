"""Exact spectra of learned copy differences; no model/data updates."""
import argparse,json,time
from pathlib import Path
import torch


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);torch.set_num_threads(16);started=time.time()
 spec={'identity':'core-difference-rank-v1','frozen_unix':time.time(),'core_layers':[0,5,10,15],
  'ranks':[4,8,16,32,64,128],'selection':'Evenly spaced fixed core layers; every attention and MLP matrix in each',
  'scope':'Read-only exact SVD of the seed23 untied terminal copy differences; no source examples, weight update or TEST. Weight-space compressibility does not establish retrained adapter performance.'}
 (a.output/'specification.json').write_text(json.dumps(spec,indent=2)+'\n');saved=torch.load(a.root/'runs/untied-depth-v1-s23-u3840/checkpoint.pt',weights_only=False,mmap=True,map_location='cpu');state=saved['model'];rows=[]
 for j in spec['core_layers']:
  prefix=f'extra_layers.{j}.'
  for key in sorted(k for k in state if k.startswith(prefix) and state[k].ndim==2):
   first=state[f'decoder.layers.{j+2}.'+key[len(prefix):]];delta=state[key]-first
   singular=torch.linalg.svdvals(delta);energy=singular.double().square();total=float(energy.sum())
   rows.append({'core_layer':j,'matrix':key[len(prefix):],'shape':list(delta.shape),
     'difference_frobenius_norm':total**.5,'energy_fraction_by_rank':{str(r):float(energy[:r].sum())/max(total,1e-30) for r in spec['ranks']},
     'rank_for_80_percent_energy':int(torch.searchsorted(energy.cumsum(0),torch.tensor(total*.8)).item()+1),
     'rank_for_95_percent_energy':int(torch.searchsorted(energy.cumsum(0),torch.tensor(total*.95)).item()+1)})
  print(json.dumps({'core_layer':j,'matrices':len(rows),'seconds':time.time()-started}),flush=True)
 (a.output/'report.json').write_text(json.dumps({'status':'COMPLETE_CORE_DIFFERENCE_SPECTRA','specification':spec,'rows':rows,'elapsed_seconds':time.time()-started},indent=2)+'\n')


if __name__=='__main__':main()
