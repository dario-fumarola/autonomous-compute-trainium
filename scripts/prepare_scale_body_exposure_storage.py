"""Fresh full raw-state admission for the prospective larger-data scope."""
import argparse
import json
from pathlib import Path
from scripts.prepare_scale_boundary_main_storage import prepare


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.root,body_exposure=True)))
