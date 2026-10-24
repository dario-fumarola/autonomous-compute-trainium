"""Smaller owned worker caps without weakening the shared host reserve."""
import ctypes
import gc
import subprocess

from .execution_isolation import metadata,verify_owner,cancel_worker_container
from .owned_native_phase import create_worker


def create_compact_worker(root,identity,half):
    # The shared factory atomically claims the half with its conservative160GiB
    # bootstrap cap. No workload starts until this owned cap is reduced to128.
    container=create_worker(root,identity,half)
    try:
        verify_owner(metadata(container),identity)
        subprocess.run(['docker','update','--memory','128g','--memory-swap','128g',container],
                       check=True,capture_output=True)
        state=metadata(container);verify_owner(state,identity)
        if state['HostConfig']['Memory']!=128*1024**3 or state['HostConfig']['MemorySwap']!=128*1024**3:
            raise ValueError('Owned128GiB cap not applied')
        return container
    except BaseException:
        cancel_worker_container(container,identity,'compact_worker_setup_failed')
        raise


def reclaim_host():
    """Return freed CPU comparison arenas before admitting another native phase."""
    gc.collect()
    ctypes.CDLL(None).malloc_trim(0)
