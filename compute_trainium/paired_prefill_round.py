"""Order actual half leases before starting a paired read-only timing round."""
import concurrent.futures as cf
import queue
import time


def paired_round(execution, phases, worker):
    """Idle Execution only; first phase gets half0, second gets half1.

    Waiting for the first actual lease avoids relying on thread submission order.
    Phase outputs must independently prove the intended physical assignment.
    Reused completed phases need no lease; their recorded assignment is checked
    by the caller. Never use this while another task shares this Execution.
    """
    if len(phases) != 2 or execution.native_active:
        raise ValueError('Exactly two phases and no active native leases required')
    if execution.halves.qsize() != 2:
        raise ValueError('Both independent halves must be idle')
    try:
        halves = [execution.halves.get_nowait(), execution.halves.get_nowait()]
    except queue.Empty:
        raise ValueError('Both independent halves must be idle') from None
    if set(halves) != {0, 1}:
        raise ValueError('Unexpected half ownership')
    for half in [0, 1]:
        execution.halves.put(half)
    with cf.ThreadPoolExecutor(2) as pool:
        first = pool.submit(worker, *phases[0])
        first.add_done_callback(lambda f: execution.failed.set()
                                if f.cancelled() or f.exception() else None)
        while not first.done():
            execution.check()
            with execution.lock:
                leased = phases[0][0] in execution.native_active
            if leased:
                break
            time.sleep(.1)
        if first.done():
            first.result()  # propagate failure before admitting its sibling
            # Reusing a completed first phase does not consume half0. Put
            # half1 first explicitly rather than accidentally rerunning on0.
            idle = [execution.halves.get_nowait(), execution.halves.get_nowait()]
            if set(idle) != {0, 1}:
                raise ValueError('Completed first phase did not return its lease')
            for half in [1, 0]:
                execution.halves.put(half)
        second = pool.submit(worker, *phases[1])
        second.add_done_callback(lambda f: execution.failed.set()
                                 if f.cancelled() or f.exception() else None)
        return [first.result(), second.result()]
