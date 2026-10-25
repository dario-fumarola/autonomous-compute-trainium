import pytest
import json
import numpy as np
from scripts.collect_low_depth_resources import union_seconds, scientific_work
from compute_trainium.training import file_sha256


def test_concurrent_intervals_are_not_charged_twice_and_gaps_remain():
    assert union_seconds([(0,5),(2,7),(7,8),(10,12),(1,3)])==10
    assert union_seconds([])==0
    with pytest.raises(ValueError):union_seconds([(2,1)])


def test_interrupted_main_replay_cost_counts_unlogged_work_and_segment_origin(tmp_path):
    experiment=tmp_path/'parcae-low-depth-pilot-v1';experiment.mkdir()
    logs=tmp_path/'logs';logs.mkdir()
    table=experiment/'work.npy';np.save(table,np.full(2100,123,dtype=np.int64))
    case={'arm':'ordinary','total_updates':2100,'work_table':table.name,
          'work_table_sha256':file_sha256(table)}
    plan={'cases':{'ordinary-main-s1231':case,'ordinary-equal-s1231':case}}
    for filename in ['main-plan.json','main-plan-recovery-v2.json']:
        (experiment/filename).write_text(json.dumps(plan))
    for identity,update,origin in [('low-depth-main-v1-ordinary-first',70,0),
                                   ('low-depth-main-r2-ordinary-finish',1950,1920),
                                   ('low-depth-main-r2-ordinary-equal',990,960)]:
        (logs/(identity+'.log')).write_text(json.dumps({'update':update})+'\n')
        bound=scientific_work(tmp_path,identity,None,0)
        assert bound['completed_optimizer_updates_lower_bound']==update-origin
        assert bound['completed_optimizer_updates_upper_bound']==update+10-origin
        assert bound['matrix_operations_lower_bound']==123*(update-origin)
        assert bound['matrix_operations_upper_bound']==123*(update+11-origin)
    report={'status':'COMPLETE_REGISTERED_LOW_DEPTH_SCIENTIFIC_SEGMENT',
            'actual_optimizer_calls_per_replica':180,'segment_real_targets':1800,
            'segment_executed_matmuls':18000}
    assert scientific_work(tmp_path,'low-depth-main-r2-ordinary-finish',report,0)['matrix_operations']==18000
