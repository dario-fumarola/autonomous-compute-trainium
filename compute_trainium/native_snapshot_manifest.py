"""Match worker validation to the unchanged owned launcher's copied trees."""
from pathlib import PurePosixPath

SNAPSHOT_TREES=frozenset({'scripts','compute_trainium','configs'})


def split_snapshot_sources(sources):
    worker,host_only={},{}
    for path,sha in sources.items():
        item=PurePosixPath(path)
        if item.is_absolute() or '..' in item.parts:
            raise ValueError('Frozen sources must be repository-relative paths')
        (worker if item.parts[0] in SNAPSHOT_TREES else host_only)[path]=sha
    return worker,host_only
