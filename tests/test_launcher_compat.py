import pytest
from compute_trainium.launcher_compat import without_launcher_corpus

def test_standard_launcher_argument_is_discarded_without_reading_or_changing_task_inputs():
    original=['--root','/work/instance-nvme','--corpus','/work/data-v1/corpus.json','--case','one']
    assert without_launcher_corpus(original)==['--root','/work/instance-nvme','--case','one']
    assert original[2:4]==['--corpus','/work/data-v1/corpus.json']
    assert without_launcher_corpus(['--case','one'])==['--case','one']
    for args in [['--corpus'],['--corpus','/different'],['--corpus','/work/data-v1/corpus.json','--corpus','/work/data-v1/corpus.json']]:
        with pytest.raises(ValueError):without_launcher_corpus(args)
