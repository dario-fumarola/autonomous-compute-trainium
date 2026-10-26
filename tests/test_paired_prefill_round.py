import queue
import threading
from types import SimpleNamespace
import pytest
from compute_trainium.paired_prefill_round import paired_round


def test_actual_leases_follow_order_and_halves_are_returned():
    halves = queue.Queue()
    for half in [1, 0]:
        halves.put(half)
    execution = SimpleNamespace(halves=halves, native_active={}, lock=threading.Lock(),
                                failed=threading.Event(), check=lambda: None)
    rendezvous = threading.Barrier(2)
    def worker(identity, value):
        half = halves.get()
        with execution.lock:
            execution.native_active[identity] = half
        rendezvous.wait(timeout=5)
        with execution.lock:
            del execution.native_active[identity]
        halves.put(half)
        return identity, half
    assert paired_round(execution, [('one', {}), ('two', {})], worker) == [('one', 0), ('two', 1)]
    assert execution.native_active == {}
    assert {halves.get_nowait(), halves.get_nowait()} == {0, 1}


def test_first_failure_does_not_admit_second_phase():
    halves = queue.Queue()
    for half in [0, 1]:
        halves.put(half)
    execution = SimpleNamespace(halves=halves, native_active={}, lock=threading.Lock(),
                                failed=threading.Event(), check=lambda: None)
    seen = []
    def worker(identity, value):
        seen.append(identity)
        raise RuntimeError('owned phase failure')
    with pytest.raises(RuntimeError, match='owned phase failure'):
        paired_round(execution, [('one', {}), ('two', {})], worker)
    assert seen == ['one']


def test_reused_first_phase_leaves_half_one_for_new_second_phase():
    halves = queue.Queue()
    for half in [0, 1]:
        halves.put(half)
    execution = SimpleNamespace(halves=halves, native_active={}, lock=threading.Lock(),
                                failed=threading.Event(), check=lambda: None)
    def worker(identity, value):
        if identity == 'reused':
            return identity, 0
        half = halves.get_nowait()
        halves.put(half)
        return identity, half
    assert paired_round(execution, [('reused', {}), ('new', {})], worker) == [('reused', 0), ('new', 1)]
