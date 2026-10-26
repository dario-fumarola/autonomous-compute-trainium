"""Timing-boundary software checks; no scientific research examples."""
import numpy as np
import pytest

from scripts.run_late_two_tune import measured_warm_seconds


def fixture():
    return dict(progress=[dict(update=u,elapsed_seconds=2*u) for u in [20,120,440,540]],
                checkpoint_records=[dict(update=u) for u in [121,432,576]])


def test_exact_warm_intervals_exclude_switch_cold_start_and_checkpoints():
    table=np.ones((577,16,32,2),dtype=np.int16);table[432:]=2
    assert measured_warm_seconds(fixture(),start=20,end=120,depth_table=table,expected_depth=1)==200
    assert measured_warm_seconds(fixture(),start=440,end=540,depth_table=table,expected_depth=2)==200
    with pytest.raises(ValueError):measured_warm_seconds(fixture(),start=440,end=540,depth_table=table,expected_depth=1)


def test_warm_interval_rejects_an_intervening_checkpoint():
    value=fixture();value['checkpoint_records'].append(dict(update=100))
    with pytest.raises(ValueError):measured_warm_seconds(value,start=20,end=120,depth_table=np.ones((577,16,32,2)),expected_depth=1)
