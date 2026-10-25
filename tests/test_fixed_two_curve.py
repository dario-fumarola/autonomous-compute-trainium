import copy
from concurrent.futures import Future
from threading import Event
import pytest
from compute_trainium.fixed_two_curve import curve_nodes,validate_curve,stop_on_failed_future


def cases():
    result={}
    for name,total in [('one-1b',1920),('fixed-1b',1920),('one-work-1b',2312),('one-2b',3840),('fixed-2b',3840),('one-work-2b',4625)]:
        arm='fixed' if name.startswith('fixed') else 'one'
        result[name]=dict(arm=arm,seed=5231,initial_state='original',peak_lr=.032,warmup_targets=50000000,depth_table_sha256=arm,work_table_sha256=arm,total_updates=total,cooldown_start_update=total//2,segment_ends=[960,1920,3840] if name.endswith('2b') and 'work' not in name else [total])
    for name,source,update in [('one-1b','one-2b',960),('fixed-1b','fixed-2b',960),('one-work-1b','one-2b',960),('one-work-2b','one-2b',1920)]:result[name]['parent']=dict(case_id=source,update=update)
    return result


def test_complete_curve_counts_shared_updates_once_and_rejects_cooled_or_changed_parents():
    c=cases();n=curve_nodes(c)
    assert validate_curve(c,n)==13657
    bad=copy.deepcopy(c);bad['one-2b']['cooldown_start_update']=959
    with pytest.raises(ValueError,match='stable prefix'):validate_curve(bad,n)
    bad=copy.deepcopy(c);bad['one-work-1b']['peak_lr']=.016
    with pytest.raises(ValueError,match='recipe'):validate_curve(bad,n)
    bad=copy.deepcopy(c);bad['fixed-1b']['parent']['case_id']='one-2b'
    with pytest.raises(ValueError,match='exact source'):validate_curve(bad,n)
    missing={k:v for k,v in n.items() if k!='fixed-1b-finish'}
    with pytest.raises(ValueError,match='finish exactly once'):validate_curve(c,missing)


def test_worker_failure_sets_paired_stop_before_main_waits_for_sibling():
    event=Event();ok=Future();ok.add_done_callback(lambda f:stop_on_failed_future(f,event));ok.set_result(None)
    assert not event.is_set()
    bad=Future();bad.add_done_callback(lambda f:stop_on_failed_future(f,event));bad.set_exception(ValueError('failed numerical gate'))
    assert event.is_set()
    event.clear();cancelled=Future();cancelled.add_done_callback(lambda f:stop_on_failed_future(f,event));cancelled.cancel()
    assert event.is_set()
