import copy
import pytest
from scripts.resume_chunked_persistent_controller import validate_launch,validate_snapshot
from compute_trainium.training import file_sha256


def test_only_the_exact_original_launch_can_be_adopted():
    p={'worker_source_sha256':{'worker.py':'source'}};options=['--plan-sha','original']
    launch=dict(entry_module='scripts.worker',arguments=options,physical_chips=list(range(16)),source_sha256=p['worker_source_sha256'])
    validate_launch(launch,p,'original','scripts.worker',options)
    for key,value in [('entry_module','scripts.other'),('arguments',['--plan-sha','changed']),
                      ('physical_chips',[0]),('source_sha256',{'worker.py':'changed'})]:
        bad=copy.deepcopy(launch);bad[key]=value
        with pytest.raises(AssertionError):validate_launch(bad,p,'original','scripts.worker',options)


def test_different_plan_cannot_reuse_matching_arguments_accidentally():
    p={'worker_source_sha256':{}}
    launch=dict(entry_module='scripts.worker',arguments=['--plan-sha','old'],physical_chips=list(range(16)),source_sha256={})
    with pytest.raises(AssertionError):validate_launch(launch,p,'new','scripts.worker',launch['arguments'])


def test_nonpython_launcher_is_verified_from_snapshot_not_python_inventory(tmp_path):
    worker=tmp_path/'worker.py';worker.write_text('original')
    launcher=tmp_path/'launch.sh';launcher.write_text('original shell')
    p={'worker_source_sha256':{f.name:file_sha256(f) for f in (worker,launcher)}}
    launch=dict(entry_module='scripts.worker',arguments=['--plan-sha','original'],
        physical_chips=list(range(16)),source_sha256={'worker.py':p['worker_source_sha256']['worker.py']})
    validate_launch(launch,p,'original','scripts.worker',launch['arguments'])
    validate_snapshot(tmp_path,p)
    launcher.write_text('changed shell')
    with pytest.raises(AssertionError):validate_snapshot(tmp_path,p)
