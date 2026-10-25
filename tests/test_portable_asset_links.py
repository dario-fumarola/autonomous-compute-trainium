"""Asset metadata must survive the host-to-worker mount-prefix change."""
import shutil
from pathlib import Path
from scripts.recover_local_batch_qualification import portable_asset_links


def test_assets_resolve_inside_the_remapped_tree(tmp_path):
    host=tmp_path/'host';previous=host/'prior';previous.mkdir(parents=True)
    (previous/'initial-state.pt').write_bytes(bytes(range(16)))
    preparation=host/'new'/'preparation';preparation.mkdir(parents=True)
    portable_asset_links(previous,preparation)
    assert not Path((preparation/'initial-state.pt').readlink()).is_absolute()
    mapped=tmp_path/'worker';shutil.copytree(host,mapped,symlinks=True)
    asset=mapped/'new/preparation/initial-state.pt'
    assert asset.resolve().is_relative_to(mapped)
    assert asset.read_bytes()==bytes(range(16))
